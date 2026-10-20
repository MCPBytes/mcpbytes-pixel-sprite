"""create_pixel_sprite, run locally or on the hosted API. Both write the same files into one folder:
raw_N.png, sprite_N.png, sprite_N.json, sprite_N.zip per candidate N, and generation.json."""

import hashlib
import json
import re
import sys
import uuid
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from mcpbytes_core.errors import ToolError
from mcpbytes_core.gpu import job_seed, stage_seed

from .inference import VERSION as INFERENCE_VERSION, inference_hash
from .models import COMPONENTS, MODEL_HASHES
from .pipeline import VERSION as PROCESSING_VERSION, package

TOOL = "create_pixel_sprite"
PALETTE = re.compile(r"#[a-fA-F0-9]{6}(,#[a-fA-F0-9]{6}){1,63}")
MAX_REFERENCE_PIXELS = 25_000_000  # the hosted tool's decoding bound


def stderr(message):
    print(message, file=sys.stderr, flush=True)


def validate(options):
    """Options with defaults applied, under the hosted tool's rules (create-pixel-sprite.ts), else invalid_options."""
    def bad(message):
        raise ToolError("invalid_options", message)

    unknown = set(options) - {"description", "variants", "transparent", "palette_colors", "palette", "frame_duration_ms", "seed"}
    if unknown:
        bad(f"unknown options: {', '.join(sorted(unknown))}")
    integer = lambda v: isinstance(v, int) and not isinstance(v, bool)  # noqa: E731
    description = options.get("description")
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 4000:
        bad("description must be 1 to 4,000 characters")
    result = {"description": description.strip(), "variants": options.get("variants", 1), "transparent": options.get("transparent", True),
              "palette_colors": options.get("palette_colors", 32), "frame_duration_ms": options.get("frame_duration_ms", 120)}
    if not integer(result["variants"]) or not 1 <= result["variants"] <= 4:
        bad("variants must be 1 to 4")
    if not isinstance(result["transparent"], bool):
        bad("transparent must be true or false")
    if result["palette_colors"] not in (16, 32, 64) or not integer(result["palette_colors"]):
        bad("palette_colors must be 16, 32 or 64")
    if not integer(result["frame_duration_ms"]) or not 40 <= result["frame_duration_ms"] <= 1000 or result["frame_duration_ms"] % 10:
        bad("frame_duration_ms must be 40 to 1,000 in steps of 10")
    if options.get("palette") is not None:
        if not isinstance(options["palette"], str) or not PALETTE.fullmatch(options["palette"]):
            bad("palette must be 2 to 64 comma-separated #RRGGBB colors without spaces")
        result["palette"] = options["palette"]
    if options.get("seed") is not None:
        if not integer(options["seed"]) or not 0 <= options["seed"] <= 0xFFFFFFFF:
            bad("seed must be an integer from 0 to 4294967295")
        result["seed"] = options["seed"]
    return result


def prepare_output(out):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ToolError("output_exists", f"{out} is not an empty folder", "choose a new output folder; results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_reference(path):
    """(RGB image at most 1024 px, its SHA-256), checked like the hosted tool's PNG/JPEG input."""
    data = Path(path).read_bytes()
    try:
        with Image.open(Path(path)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise ToolError("invalid_image", "the reference must be a PNG or JPEG")
            if image.width * image.height > MAX_REFERENCE_PIXELS:
                raise ToolError("invalid_image", "the reference must have at most 25 million pixels")
            reference = image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as e:
        raise ToolError("invalid_image", f"cannot read the reference image: {e}") from None
    reference.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return reference, hashlib.sha256(data).hexdigest()


def run_local(options, reference=None, out=".", engine=None, offload="auto", progress=stderr):
    """Generate on this machine's GPU. Long-lived callers (the MCP server) pass a loaded `engine` to reuse."""
    options = validate(options)
    seed = job_seed(options.get("seed"))
    image, reference_hash = load_reference(reference) if reference else (None, None)
    out = prepare_output(out)
    if engine is None:
        from .engine import Engine

        progress("Loading models")
        engine = Engine(offload)
    provenance = {"inference_version": INFERENCE_VERSION, "inference_sha256": inference_hash(bool(reference)),
                  "model_revision": COMPONENTS["main"]["revision"], "reference_sha256": reference_hash,
                  "source_revisions": [COMPONENTS[c]["revision"] for c in ("lora", "encoder")], "model_hashes": MODEL_HASHES,
                  "runtime": f"mcpbytes-pixel-sprite {version('mcpbytes-pixel-sprite')}"}
    stages = []
    for number in range(1, options["variants"] + 1):
        progress(f"Creating sprite {number} of {options['variants']}")
        stage = {**options, "variant": number, "seed": stage_seed(seed, f"sprite_{number}")}
        raw = engine.render(options["description"], image, stage["seed"])
        stages.append(package(raw, out, stage, provenance))
    settings = {k: v for k, v in options.items() if k != "description"}  # prompts stay out of generation.json, as in the cloud
    (out / "generation.json").write_text(json.dumps({
        "tool": TOOL, "runtime": "local", "options": {**settings, "seed": seed}, "inference_version": INFERENCE_VERSION,
        "processing_version": PROCESSING_VERSION, "model_revisions": {c: s["revision"] for c, s in COMPONENTS.items()},
        "stages": [{"stage_id": f"sprite_{s['variant']}", **s} for s in stages]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summarize(out, stages, seed=seed)


def run_cloud(options, reference=None, out=".", cloud=None, idempotency_key=None, progress=stderr):
    """Generate on the hosted API for credits. Reuse `idempotency_key` after a lost response: no second charge."""
    from mcpbytes_core.cloud import Cloud

    options = validate(options)
    cloud = cloud or Cloud()
    out = prepare_output(out)
    key = idempotency_key or str(uuid.uuid4())
    progress(f"Request key {key}: reuse it with --idempotency-key if this run is interrupted")
    upload_id = cloud.upload(reference) if reference else None
    job = cloud.start(TOOL, options, key, upload_id)
    progress(f"Job {job['id']}")
    job = cloud.wait(job, lambda j: progress((j.get("workflow") or {}).get("stage") or j["status"]))
    cloud.download(job, out)
    stages = [json.loads((out / f"sprite_{n}.json").read_text(encoding="utf-8")) for n in range(1, options["variants"] + 1)]
    return summarize(out, stages, job_id=job["id"], credits_charged=job.get("credits_charged"))


def summarize(out, stages, **extra):
    return {"qa_status": "needs_review" if any(s["qa_status"] == "needs_review" for s in stages) else "unreviewed",
            "output_dir": str(out.resolve()), "files": sorted(p.name for p in out.iterdir()),
            "variants": [{"variant": s["variant"], "sheet": f"sprite_{s['variant']}.png", "archive": f"sprite_{s['variant']}.zip",
                          "qa_status": s["qa_status"], "warnings": s["warnings"]} for s in stages],
            "notes": ["Review anatomy, direction, identity and transparency before using these sprites."], **extra}
