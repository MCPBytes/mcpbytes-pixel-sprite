"""Versioned prompt/layout and fixed direct-pipeline settings, not caller-selected models."""

import hashlib
import json

VERSION = "sprite_diffusers_v1"
LAYOUT_PROMPT = (
    "Four rows, four equal cells per row. Row one: three walking poses facing down, then arms raised. "
    "Row two: three walking poses facing left, then a jumping-left still pose. "
    "Row three: three walking poses facing right, then a jumping-right still pose. "
    "Row four: three walking poses facing up, then a lying still pose. "
    "One complete character in every cell, consistent identity and clothing. Leave space around every silhouette. "
    "Plain solid magenta background (#ff00ff), no checkerboard, no lettering, no grid lines, no shadows. "
    "Do not use magenta in the character."
)


def request_options(description, reference=None):
    lead = "Pixel art spritesheet of the character in the reference. " if reference is not None else "Pixel art character spritesheet. "
    return {"prompt": lead + description + ". " + LAYOUT_PROMPT, "image": reference,
            "width": 512, "height": 512, "num_inference_steps": 20, "guidance_scale": 5.0, "num_images_per_prompt": 1}


def inference_hash(reference=False):
    settings = request_options("", "reference" if reference else None)
    return hashlib.sha256(json.dumps({"version": VERSION, **settings}, sort_keys=True).encode()).hexdigest()
