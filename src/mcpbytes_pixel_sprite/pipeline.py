"""CPU-only sprite exports. No learned segmentation, recentering, or invented motion."""

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from mcpbytes_core.errors import ToolError
from .engine_exports import engine_exports

VERSION = "sprite_processing_v3"
LAYOUT = "small_character_32_v1"
DIRECTIONS = ("down", "left", "right", "up")
POSES = ("arms_raised_pose", "jump_left_pose", "jump_right_pose", "lying_pose")
WALK = (0, 1, 2, 1)
KEY = (255, 0, 255)


def encoded(image, **kwargs):
    target = io.BytesIO()
    image.save(target, **kwargs)
    return target.getvalue()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def downscale(image, method="nearest"):
    if image.size != (512, 512):
        raise ToolError("invalid_output", "sprite render must be exactly 512 by 512")
    rgba = image.convert("RGBA")
    if method == "nearest":
        return rgba.resize((128, 128), Image.Resampling.NEAREST)
    if method != "centroid":
        raise ValueError("unknown downscale method")
    # Two deterministic clusters per 4x4 block. Qualification compares this candidate with nearest;
    # the dominant centroid preserves crisp edges without averaging foreground into background.
    output = Image.new("RGBA", (128, 128))
    pixels = rgba.load()
    for y in range(128):
        for x in range(128):
            values = [pixels[x * 4 + dx, y * 4 + dy] for dy in range(4) for dx in range(4)]
            centers = [min(values), max(values)]
            groups = [values, []]
            for _ in range(4):
                groups = [[], []]
                for value in values:
                    choice = min(range(2), key=lambda k: sum((a - b) ** 2 for a, b in zip(value, centers[k])))
                    groups[choice].append(value)
                centers = [tuple(round(sum(v[c] for v in group) / len(group)) for c in range(4)) if group else centers[k] for k, group in enumerate(groups)]
            output.putpixel((x, y), centers[0 if len(groups[0]) >= len(groups[1]) else 1])
    return output


def background(image, transparent):
    """Remove the reserved key, then unmix key spill at silhouette edges before palette selection."""
    image = image.convert("RGBA")
    if not transparent:
        opaque = Image.new("RGBA", image.size, (*KEY, 255))
        opaque.alpha_composite(image)
        return opaque, [], "opaque"
    if image.getchannel("A").getextrema()[0] < 255:
        # Pixel art uses binary alpha, consistently across every cell.
        image.putalpha(image.getchannel("A").point(lambda a: 255 if a >= 128 else 0))
        return image, [], "source_alpha"
    pixels = image.load()
    width, height = image.size
    # Inference does not reproduce a hex value exactly. Accept its observed key only when
    # the border is overwhelmingly uniform and still magenta; a gray checkerboard is not alpha.
    border = [pixels[x, y][:3] for y in range(height) for x in range(width) if x in (0, width - 1) or y in (0, height - 1)]
    observed = Counter(border).most_common(1)[0][0]
    near = lambda color, key: max(abs(color[c] - key[c]) for c in range(3)) <= 24
    uniform = sum(near(color, observed) for color in border) >= len(border) * 0.9
    magenta = observed[0] >= 192 and observed[1] <= 64 and observed[2] >= 128
    warnings = ["background_mask_needs_review"]
    if not uniform or not magenta:
        return image, warnings + ["background_not_removed"], "unrecognized_background"
    key = observed
    # The fixed prompt reserves magenta for the background. Mask matching pixels even in
    # enclosed arm/leg gaps; connectivity alone left bright pink islands in finished sheets.
    removed = {(x, y) for y in range(height) for x in range(width)
               if max(abs(pixels[x, y][c] - key[c]) for c in range(3)) <= 48}
    original = image.copy().load()
    def neighbors(x, y, radius=1):
        return ((nx, ny) for ny in range(max(0, y - radius), min(height, y + radius + 1))
                for nx in range(max(0, x - radius), min(width, x + radius + 1)) if (nx, ny) != (x, y))
    # A contaminated edge C is a mixture (1-b)*foreground + b*key. Estimate foreground
    # from nearby non-key colours and remove its key contribution. This preserves opaque,
    # binary pixel-art edges; a CSS filter or palette-wide colour replacement cannot do so.
    for y in range(height):
        for x in range(width):
            if (x, y) in removed or not any(p in removed for p in neighbors(x, y)):
                continue
            color = original[x, y][:3]
            if min(color[0], color[2]) - color[1] <= 24:
                continue  # Natural skin, red clothing, neutrals and green details stay unchanged.
            fits = []
            for nx, ny in neighbors(x, y, 2):
                foreground = original[nx, ny][:3]
                if (nx, ny) in removed or min(foreground[0], foreground[2]) - foreground[1] > 24:
                    continue
                vector = [key[c] - foreground[c] for c in range(3)]
                norm = sum(v * v for v in vector)
                if not norm:
                    continue
                blend = sum((color[c] - foreground[c]) * vector[c] for c in range(3)) / norm
                residual = sum((color[c] - (foreground[c] + blend * vector[c])) ** 2 for c in range(3))
                if 0.1 <= blend <= 1 and residual <= 3 * 24 ** 2:
                    fits.append((residual + 4 * ((nx - x) ** 2 + (ny - y) ** 2), blend))
            if fits:
                _, blend = min(fits)
                if blend >= 0.5:
                    pixels[x, y] = (0, 0, 0, 0)
                else:
                    pixels[x, y] = (*[max(0, min(255, round((color[c] - blend * key[c]) / (1 - blend)))) for c in range(3)], 255)
    for point in removed:
        pixels[point] = (0, 0, 0, 0)
    return image, warnings, "reserved_chroma_key_despill"


def quantize(image, colors=32, supplied=None):
    visible = [pixel[:3] for pixel in image.get_flattened_data() if pixel[3]]
    if not visible:
        raise ToolError("invalid_output", "sprite sheet has no visible content")
    if supplied:
        palette = [tuple(bytes.fromhex(color[1:])) for color in supplied.split(",")]
        if not 2 <= len(palette) <= 64 or any(len(c) != 3 for c in palette):
            raise ValueError("invalid palette")
    else:
        sample = Image.new("RGB", (len(visible), 1))
        sample.putdata(visible)
        reduced = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        palette = sorted(set(reduced.convert("RGB").get_flattened_data()))
    # Map once per distinct colour, so alpha/background and per-frame content never alter the palette.
    mapping = {color: min(palette, key=lambda p: sum((a - b) ** 2 for a, b in zip(color, p))) for color in set(visible)}
    result = Image.new("RGBA", image.size)
    result.putdata([(*mapping[p[:3]], 255) if p[3] else (0, 0, 0, 0) for p in image.get_flattened_data()])
    return result, palette


def inspect_sheet(sheet, initial_warnings):
    warnings = list(initial_warnings)
    cells = []
    for row in range(4):
        for column in range(4):
            frame = sheet.crop((column * 32, row * 32, (column + 1) * 32, (row + 1) * 32))
            alpha = frame.getchannel("A")
            bbox = alpha.getbbox()
            count = sum(1 for a in alpha.get_flattened_data() if a)
            flags = []
            if count < 8:
                flags.append("empty_or_tiny_cell")
            if count > 900:
                flags.append("background_or_overfilled_cell")
            if bbox and (bbox[0] == 0 or bbox[1] == 0 or bbox[2] == 32 or bbox[3] == 32):
                flags.append("cell_edge_contact")
            colors = len({p[:3] for p in frame.get_flattened_data() if p[3]})
            cells.append({"index": row * 4 + column, "bbox": list(bbox) if bbox else None, "occupied_pixels": count, "colors": colors, "warnings": flags})
            warnings.extend(f"cell_{row * 4 + column}:{flag}" for flag in flags)
    for row in range(4):
        walk = cells[row * 4:row * 4 + 3]
        sizes = [cell["occupied_pixels"] for cell in walk]
        if min(sizes) and max(sizes) > min(sizes) * 2:
            warnings.append(f"walk_{DIRECTIONS[row]}:occupancy_change")
        counts = [cell["colors"] for cell in walk]
        if min(counts) and max(counts) > min(counts) * 2:
            warnings.append(f"walk_{DIRECTIONS[row]}:palette_change")
    return {"qa_status": "needs_review" if warnings else "unreviewed", "warnings": sorted(set(warnings)), "cells": cells,
            "human_review": ["anatomy", "identity_across_views", "direction_and_pose", "background_and_thin_details"]}


def preview(frames, palette, duration):
    """One GIF palette, transparent index 0, restore-to-background disposal, infinite loop."""
    flat = [0, 0, 0] + [value for color in palette for value in color]
    flat += [0] * (768 - len(flat))
    lookup = {color: i + 1 for i, color in enumerate(palette)}
    indexed = []
    for frame in frames:
        target = Image.new("P", frame.size)
        target.putpalette(flat)
        target.putdata([lookup[p[:3]] if p[3] else 0 for p in frame.get_flattened_data()])
        indexed.append(target)
    return encoded(indexed[0], format="GIF", save_all=True, append_images=indexed[1:], duration=duration, loop=0, transparency=0, disposal=2, optimize=False)


def package(raw, out, options, provenance, *, method="nearest"):
    """Write four immutable stage outputs; all sixteen fixed cells remain available even with QA flags."""
    out = Path(out)
    number = options.get("variant", 1)
    raw_name, sheet_name, meta_name, zip_name = f"raw_{number}.png", f"sprite_{number}.png", f"sprite_{number}.json", f"sprite_{number}.zip"
    # Raw bytes precede lossy processing. Save without prompt-bearing image metadata.
    raw_bytes = encoded(raw.convert("RGBA"), format="PNG")
    (out / raw_name).write_bytes(raw_bytes)
    native = downscale(raw, method)
    masked, warnings, mask_method = background(native, options.get("transparent", True))
    sheet, palette = quantize(masked, options.get("palette_colors", 32), options.get("palette"))
    sheet_bytes = encoded(sheet, format="PNG")
    (out / sheet_name).write_bytes(sheet_bytes)
    qa = inspect_sheet(sheet, warnings)
    frames, entries, rectangles = [], {"raw.png": raw_bytes, "sheet.png": sheet_bytes}, []
    for row in range(4):
        for column in range(4):
            name = f"walk_{DIRECTIONS[row]}_{column}" if column < 3 else POSES[row]
            rect = [column * 32, row * 32, 32, 32]
            frame = sheet.crop((rect[0], rect[1], rect[0] + 32, rect[1] + 32))
            frames.append(frame)
            filename = f"frames/{name}.png"
            entries[filename] = encoded(frame, format="PNG")
            rectangles.append({"id": name, "file": filename, "rect": rect, "size": [32, 32], "pivot": [16, 32]})
    duration = options.get("frame_duration_ms", 120)
    animations = {}
    for row, direction in enumerate(DIRECTIONS):
        sequence = [row * 4 + n for n in WALK]
        filename = f"previews/walk_{direction}.gif"
        entries[filename] = preview([frames[index] for index in sequence], palette, duration)
        animations[f"walk_{direction}"] = {"frames": [rectangles[index]["id"] for index in sequence], "durations_ms": [duration] * 4, "loop": True, "preview": filename, "unique_poses": 3}
    palette_hex = ["#" + bytes(color).hex() for color in palette]
    manifest = {"schema_version": 1, "layout_version": LAYOUT, "processing_version": VERSION,
                "sheet": "sheet.png", "size": [128, 128], "cell_size": [32, 32], "frames": rectangles,
                "animations": animations, "palette": palette_hex, "palette_id": hashlib.sha256(json_bytes(palette_hex)).hexdigest(),
                "seed": options["seed"], "variant": number, "transparent": sheet.getchannel("A").getextrema()[0] == 0,
                "downscale": method, "background_method": mask_method, "qa_status": qa["qa_status"], "warnings": qa["warnings"],
                "engine_exports": {"godot": "godot/character.tscn", "unity": "unity/import.json"}}
    entries.update(engine_exports(manifest))
    entries["manifest.json"] = json_bytes(manifest)
    entries["qa.json"] = json_bytes(qa)
    entries["provenance.json"] = json_bytes({**provenance, "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(), "processing_version": VERSION})
    (out / meta_name).write_bytes(json_bytes({**manifest, "sheet": sheet_name, "archive": zip_name, "qa": qa}))
    # Stable timestamps, ordering, modes and compression make exports reproducible from the same render.
    with zipfile.ZipFile(out / zip_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    return {"qa_status": qa["qa_status"], "warnings": qa["warnings"], "variant": number, "width": 128, "height": 128,
            "frame_count": 16, "walk_poses_per_direction": 3, "seed": options["seed"], "processing_version": VERSION}
