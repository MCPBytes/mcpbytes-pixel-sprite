"""Deterministic engine adapters from the same rectangles/timing used by PNG/GIF exports."""

import json
from pathlib import Path


def engine_exports(manifest):
    frames, animations = manifest["frames"], manifest["animations"]
    ids = {frame["id"]: index for index, frame in enumerate(frames)}
    resource = [f'[gd_resource type="SpriteFrames" load_steps={len(frames) + 2} format=3]',
                '', '[ext_resource type="Texture2D" path="../sheet.png" id="1"]']
    for index, frame in enumerate(frames):
        rect = ", ".join(str(n) for n in frame["rect"])
        resource.extend(['', f'[sub_resource type="AtlasTexture" id="Frame_{index}"]',
                         'atlas = ExtResource("1")', f'region = Rect2({rect})', 'filter_clip = true'])
    definitions = []
    for name, animation in animations.items():
        base = animation["durations_ms"][0]
        sequence = ',\n'.join('{"duration": %.12g, "texture": SubResource("Frame_%d")}' % (duration / base, ids[frame])
                              for frame, duration in zip(animation["frames"], animation["durations_ms"]))
        definitions.append('{"frames": [\n%s\n], "loop": %s, "name": &%s, "speed": %.12g}' %
                           (sequence, str(animation["loop"]).lower(), json.dumps(name), 1000 / base))
    resource.extend(['', '[resource]', 'animations = [\n' + ',\n'.join(definitions) + '\n]'])
    pivot = frames[0]["pivot"]
    scene = f'''[gd_scene load_steps=2 format=3]

[ext_resource type="SpriteFrames" path="sprite_frames.tres" id="1"]

[node name="PixelCharacter" type="AnimatedSprite2D"]
texture_filter = 1
sprite_frames = ExtResource("1")
animation = &"walk_down"
autoplay = "walk_down"
centered = false
offset = Vector2({-pivot[0]}, {-pivot[1]})
'''
    height = manifest["size"][1]
    unity_frames = []
    for frame in frames:
        x, y, w, h = frame["rect"]
        unity_frames.append({"name": frame["id"], "rect": {"x": x, "y": height - y - h, "width": w, "height": h},
                             "pivot": {"x": frame["pivot"][0] / w, "y": 1 - frame["pivot"][1] / h}})
    unity = {"schema_version": 1, "sheet": "../sheet.png", "width": manifest["size"][0], "height": height,
             "pixels_per_unit": manifest["cell_size"][1], "frames": unity_frames,
             "animations": [{"name": name, "frames": animation["frames"], "durations_ms": animation["durations_ms"],
                             "loop": animation["loop"]} for name, animation in animations.items()]}
    return {
        "godot/sprite_frames.tres": ("\n".join(resource) + "\n").encode(),
        "godot/character.tscn": scene.encode(),
        "unity/import.json": (json.dumps(unity, indent=2, allow_nan=False) + "\n").encode(),
        # Keep the helper as text in each archive: multiple candidates must not install duplicate C# classes.
        "unity/MCPBytesSpriteImporter.cs.txt": (Path(__file__).parent / "templates/MCPBytesSpriteImporter.cs.txt").read_bytes(),
        "IMPORTING.md": (Path(__file__).parent / "templates/IMPORTING.md").read_bytes(),
    }
