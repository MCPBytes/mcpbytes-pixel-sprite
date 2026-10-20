"""Fixture-based export/QA and fixed-workflow contracts. These do not qualify model quality."""

import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image, ImageDraw

from mcpbytes_core.errors import ToolError
from mcpbytes_pixel_sprite.pipeline import background, downscale, package, quantize
from mcpbytes_pixel_sprite.inference import inference_hash, request_options


def fixture():
    sheet = Image.new("RGBA", (128, 128), (255, 0, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row in range(4):
        for col in range(4):
            x, y = col * 32, row * 32
            draw.rectangle((x + 10, y + 8, x + 20, y + 27), fill=(30, 110, 45, 255))
            draw.rectangle((x + 12, y + 5, x + 18, y + 12), fill=(240, 225, 210, 255))
            draw.point((x + 10 + col, y + 25), fill=(60, 35, 20, 255))
    return sheet.resize((512, 512), Image.Resampling.NEAREST)


def options(**kwargs):
    return {"seed": 42, "variant": 1, "transparent": True, "palette_colors": 32, "frame_duration_ms": 120, **kwargs}


def test_exports_are_deterministic_and_frames_reconstruct_sheet(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    summary = package(fixture(), a, options(), {"workflow_sha256": "a" * 64})
    package(fixture(), b, options(), {"workflow_sha256": "a" * 64})
    assert summary["qa_status"] == "needs_review"  # automatic masking never means artist approval
    for file in a.iterdir():
        assert file.read_bytes() == (b / file.name).read_bytes()
    with zipfile.ZipFile(a / "sprite_1.zip") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert len(manifest["frames"]) == 16
        rebuilt = Image.new("RGBA", (128, 128))
        for frame in manifest["frames"]:
            image = Image.open(io.BytesIO(archive.read(frame["file"])))
            assert image.size == (32, 32)
            assert frame["pivot"] == [16, 32]
            rebuilt.paste(image, frame["rect"][:2])
        assert rebuilt.tobytes() == Image.open(a / "sprite_1.png").tobytes()
        assert manifest["frames"][15]["id"] == "lying_pose"
        assert all(animation["unique_poses"] == 3 for animation in manifest["animations"].values())
        provenance = json.loads(archive.read("provenance.json"))
        assert provenance["raw_sha256"] == hashlib.sha256(archive.read("raw.png")).hexdigest()


def test_gif_loops_have_exact_timing_transparency_and_disposal(tmp_path):
    package(fixture(), tmp_path, options(frame_duration_ms=80), {})
    with zipfile.ZipFile(tmp_path / "sprite_1.zip") as archive:
        for direction in ("down", "left", "right", "up"):
            with Image.open(io.BytesIO(archive.read(f"previews/walk_{direction}.gif"))) as gif:
                assert gif.info["loop"] == 0
                assert gif.n_frames == 4
                frames = []
                for i in range(4):
                    gif.seek(i)
                    assert gif.info["duration"] == 80
                    assert gif.disposal_method == 2
                    frame = gif.convert("RGBA")
                    assert frame.getpixel((0, 0))[3] == 0
                    frames.append(frame.tobytes())
                assert frames[1] == frames[3]


def test_shared_supplied_palette_and_binary_alpha(tmp_path):
    package(fixture(), tmp_path, options(palette="#ffffff,#000000,#008000"), {})
    image = Image.open(tmp_path / "sprite_1.png")
    assert {p[:3] for p in image.get_flattened_data() if p[3]} <= {(255, 255, 255), (0, 0, 0), (0, 128, 0)}
    assert {p[3] for p in image.get_flattened_data()} == {0, 255}


def test_engine_exports_share_frame_geometry_timing_and_safe_installation(tmp_path):
    package(fixture(), tmp_path, options(frame_duration_ms=80), {})
    with zipfile.ZipFile(tmp_path / "sprite_1.zip") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        unity = json.loads(archive.read("unity/import.json"))
        godot = archive.read("godot/sprite_frames.tres").decode()
        scene = archive.read("godot/character.tscn").decode()
        assert manifest["engine_exports"] == {"godot": "godot/character.tscn", "unity": "unity/import.json"}
        assert len(unity["frames"]) == len(manifest["frames"]) == 16
        assert unity["frames"][0]["rect"] == {"x": 0, "y": 96, "width": 32, "height": 32}
        assert unity["frames"][-1]["rect"]["y"] == 0
        assert all(frame["pivot"] == {"x": 0.5, "y": 0.0} for frame in unity["frames"])
        assert all(motion["durations_ms"] == [80] * 4 and motion["loop"] for motion in unity["animations"])
        assert '"speed": 12.5' in godot and godot.count('[sub_resource type="AtlasTexture"') == 16
        assert 'path="../sheet.png"' in godot
        assert 'offset = Vector2(-16, -32)' in scene and 'texture_filter = 1' in scene
        assert "unity/MCPBytesSpriteImporter.cs.txt" in archive.namelist()
        assert not any(name.endswith(".cs") for name in archive.namelist())  # No duplicate classes when several ZIPs are extracted.
        assert "IMPORTING.md" in archive.namelist()
        assert len(archive.namelist()) == 30


def test_mask_preserves_light_clothing_and_removes_enclosed_reserved_key():
    image = Image.new("RGBA", (12, 12), (255, 0, 255, 255))
    ImageDraw.Draw(image).rectangle((3, 3, 8, 8), fill="white")
    image.putpixel((5, 5), (255, 0, 255, 255))
    result, warnings, _ = background(image, True)
    assert result.getpixel((3, 3)) == (255, 255, 255, 255)
    assert result.getpixel((5, 5))[3] == 0
    assert result.getpixel((0, 0))[3] == 0
    assert "background_mask_needs_review" in warnings


def test_key_spill_is_unmixed_before_palette_selection_without_recoloring_foreground():
    key, green = (253, 16, 179), (30, 80, 20)
    image = Image.new("RGBA", (32, 32), (*key, 255))
    ImageDraw.Draw(image).rectangle((8, 8, 23, 23), fill=(*green, 255))
    blend = lambda amount: (*[round(amount * key[c] + (1 - amount) * green[c]) for c in range(3)], 255)
    image.putpixel((8, 10), blend(0.4))  # Visible pink spill, mostly foreground: retain the edge.
    image.putpixel((7, 10), blend(0.75))  # Mostly background: make it transparent.
    for point, color in [((8, 15), (220, 130, 150, 255)), ((8, 20), (200, 40, 60, 255)), ((14, 14), (90, 50, 120, 255))]:
        image.putpixel(point, color)
    result, _, method = background(image, True)
    assert method == "reserved_chroma_key_despill"
    assert result.getpixel((7, 10)) == (0, 0, 0, 0)
    assert result.getpixel((8, 10))[3] == 255
    assert max(abs(a - b) for a, b in zip(result.getpixel((8, 10))[:3], green)) <= 2
    for point in ((8, 15), (8, 20), (14, 14), (15, 15)):
        assert result.getpixel(point) == image.getpixel(point)
    indexed, palette = quantize(result)
    assert key not in palette
    assert indexed.getpixel((7, 10))[3] == 0


def test_unrecognized_background_does_not_erase_magenta_character_details():
    image = Image.new("RGBA", (32, 32), "cyan")
    ImageDraw.Draw(image).rectangle((8, 8, 23, 23), fill="magenta")
    result, warnings, _ = background(image, True)
    assert result.tobytes() == image.tobytes()
    assert "background_not_removed" in warnings


def test_mask_uses_a_uniform_generated_magenta_shade_without_removing_foreground():
    image = Image.new("RGBA", (32, 32), (250, 15, 205, 255))
    ImageDraw.Draw(image).rectangle((8, 8, 23, 23), fill=(245, 245, 245, 255))
    result, warnings, _ = background(image, True)
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((10, 10)) == (245, 245, 245, 255)
    assert "background_not_removed" not in warnings
    assert "background_mask_needs_review" in warnings


def test_checkerboard_is_not_mistaken_for_transparency(tmp_path):
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, 512, 16):
        for x in range(0, 512, 16):
            if (x + y) // 16 % 2:
                draw.rectangle((x, y, x + 15, y + 15), fill="#cccccc")
    result = package(image, tmp_path, options(), {})
    assert "background_not_removed" in result["warnings"]
    assert result["qa_status"] == "needs_review"
    assert Image.open(tmp_path / "sprite_1.png").getchannel("A").getextrema() == (255, 255)


def test_bad_dimensions_and_blank_sheets_fail(tmp_path):
    with pytest.raises(ToolError, match="512"):
        package(Image.new("RGBA", (128, 128)), tmp_path, options(), {})
    with pytest.raises(ToolError, match="visible"):
        package(Image.new("RGBA", (512, 512), (255, 0, 255, 255)), tmp_path, options(), {})


def test_boundary_clipping_and_empty_cells_are_reported(tmp_path):
    image = fixture()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 127, 127), fill=(255, 0, 255, 255))
    draw.rectangle((128, 30, 190, 110), fill="white")
    summary = package(image, tmp_path, options(), {})
    assert "cell_0:empty_or_tiny_cell" in summary["warnings"]
    assert "cell_1:cell_edge_contact" in summary["warnings"]


def test_downscale_candidates_are_deterministic():
    source = fixture()
    assert downscale(source, "centroid").tobytes() == downscale(source, "nearest").tobytes()
    assert downscale(source, "centroid").tobytes() == downscale(source, "centroid").tobytes()


def test_direct_inference_keeps_dimensions_guidance_and_motion_fixed():
    text = request_options('a ranger "}; malicious workflow')
    reference = Image.new("RGB", (32, 32))
    edit = request_options("a ranger", reference)
    assert text["image"] is None and edit["image"] is reference
    assert text["width"] == text["height"] == 512
    assert text["num_inference_steps"] == 20 and text["guidance_scale"] == 5.0
    assert text["num_images_per_prompt"] == 1
    assert "malicious workflow" in text["prompt"]
    assert inference_hash() != inference_hash(True)
