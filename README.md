# mcpbytes-pixel-sprite

Generate pixel-art character sprite sheets on your own NVIDIA GPU, from the command line or from any MCP
client (Claude, Cursor, VS Code, …). It is the open-source engine behind MCPBytes' hosted
`create_pixel_sprite` tool, and can also send jobs to the hosted tool when you have no GPU.

Each candidate is a 128×128 PNG of sixteen 32×32 cells. Rows face down, left, right and up; columns 0–2
are walk poses and column 3 holds four still poses (arms raised, jump left, jump right, lying). Every
candidate also comes with frame PNGs, looping walk GIFs, QA metadata, a Godot 4 scene and a Unity 6 importer.

## Examples

Real, unedited outputs of the hosted `create_pixel_sprite` tool, which runs this engine (one variant, seed 42), enlarged
here with square pixels. More on the [project page](https://mcpbytes.github.io/mcpbytes-pixel-sprite/).

| Sheet | Walk | Prompt |
|---|---|---|
| <img src="https://raw.githubusercontent.com/MCPBytes/mcpbytes-pixel-sprite/main/docs/examples/readme/wizard-2x.png" width="256" height="256" alt="An old wizard in a violet star-embroidered robe with a bent pointed hat and a teal crystal staff, in sixteen pixel-art poses"> | <img src="https://raw.githubusercontent.com/MCPBytes/mcpbytes-pixel-sprite/main/docs/examples/readme/wizard-walk-right-4x.gif" width="128" height="128" alt="Animated wizard walking to the right"> | An ancient wizard with a long braided white beard, a deep violet robe embroidered with silver stars, a wide-brimmed pointed hat bent at the tip, a gnarled oak staff crowned with a glowing teal crystal, and a leather spellbook hanging from a rope belt. |
| <img src="https://raw.githubusercontent.com/MCPBytes/mcpbytes-pixel-sprite/main/docs/examples/readme/robot-2x.png" width="256" height="256" alt="A junkyard robot with a rusty orange drum body, one round camera eye and a bent antenna, in sixteen pixel-art poses"> | <img src="https://raw.githubusercontent.com/MCPBytes/mcpbytes-pixel-sprite/main/docs/examples/readme/robot-walk-down-4x.gif" width="128" height="128" alt="Animated junkyard robot walking down"> | A scrappy junkyard robot built from a rusty orange oil drum body, mismatched green and grey arms, a single big round camera eye with a blue lens, a bent antenna with a red bulb, tank treads for feet, and exposed copper wires. |
| <img src="https://raw.githubusercontent.com/MCPBytes/mcpbytes-pixel-sprite/main/docs/examples/readme/pirate-2x.png" width="256" height="256" alt="A pirate captain in a gold-trimmed tricorn hat and scarlet coat with a cutlass, in sixteen pixel-art poses"> | <img src="https://raw.githubusercontent.com/MCPBytes/mcpbytes-pixel-sprite/main/docs/examples/readme/pirate-walk-right-4x.gif" width="128" height="128" alt="Animated pirate captain walking to the right"> | A swaggering pirate captain with a black tricorn hat trimmed in gold, a long scarlet coat with brass buttons, a striped shirt, a peg leg, a hook hand, a curved cutlass at the hip and a bright green parrot perched on the shoulder. |
| <img src="https://raw.githubusercontent.com/MCPBytes/mcpbytes-pixel-sprite/main/docs/examples/readme/astronaut-2x.png" width="256" height="256" alt="An astronaut in a white spacesuit with a gold visor and a grey backpack, in sixteen pixel-art poses"> | <img src="https://raw.githubusercontent.com/MCPBytes/mcpbytes-pixel-sprite/main/docs/examples/readme/astronaut-walk-down-4x.gif" width="128" height="128" alt="Animated astronaut walking down"> | A cheerful astronaut in a bulky white spacesuit with orange stripes, a domed helmet with a reflective gold visor, a chunky grey life-support backpack with two small thrusters, mission patches on the chest and thick padded boots. |

## Requirements

- An NVIDIA GPU with a CUDA 12.8 driver (R570 or newer) on Windows or Linux. The memory mode is picked
  from the GPU's size; you can override it with `--offload`:

  | `--offload` | GPU memory used | Time per candidate* |
  |---|---|---|
  | `none` (picked for GPUs with 20 GB or more) | 16.2 GiB | 2.3 s |
  | `model` (picked for 10–20 GB) | 7.8 GiB | 7.4 s |
  | `sequential` (picked for less than 10 GB) | 1.0 GiB | 34 s |

  \*Measured on an RTX PRO 6000 with the models already loaded. Your times will differ.
- About 16 GB of disk for the model files, downloaded once into the Hugging Face cache.
- [uv](https://docs.astral.sh/uv/) (or Python 3.11–3.13 with pip).
- No NVIDIA GPU (an Apple Silicon Mac, for example)? Use `--cloud` with an MCPBytes API key; see below.

## Install and run

```bash
uvx --torch-backend=auto mcpbytes-pixel-sprite download
uvx --torch-backend=auto mcpbytes-pixel-sprite generate "A forest ranger with a green hood and a brown satchel" --seed 42
```

`--torch-backend=auto` installs the CUDA build of PyTorch that matches your driver. Without it, PyTorch from
PyPI is CPU-only on Windows. With pip: `pip install mcpbytes-pixel-sprite --extra-index-url https://download.pytorch.org/whl/cu128`.

`download` fetches the pinned files listed below (about 16 GB, resumable). `generate` also downloads anything
that is missing, then writes to `pixel-sprites/<timestamp>/` (or to `-o <new or empty folder>`):

| File | Contents |
|---|---|
| `raw_N.png` | The model's 512×512 render, before any processing |
| `sprite_N.png` | The 128×128 sheet: transparent background, one shared palette |
| `sprite_N.json` | Frame rectangles and pivots, animations, palette, QA status and warnings |
| `sprite_N.zip` | Sheet, 16 frames, 4 walk GIFs, `manifest.json`, `qa.json`, `provenance.json`, `godot/`, `unity/`, `IMPORTING.md` |
| `generation.json` | Effective options, seeds and model revisions (your description is left out) |

Options: `--variants 1-4`, `--reference image.png` (a PNG or JPEG of your character), `--opaque`,
`--palette-colors 16|32|64`, `--palette "#rrggbb,#rrggbb,…"`, `--frame-duration-ms 40-1000`, `--seed`. The same
seed, inputs, hardware and `--offload` mode reproduce a result byte for byte. Seeds are derived the same way as
in the hosted tool, but individual pixels can still differ between GPUs.

`qa_status` is `unreviewed` or `needs_review`; it is never an approval. Check anatomy, facing direction,
identity across views and transparency yourself. Walk previews play poses `[0, 1, 2, 1]`: three drawn poses,
not four.

## MCP server

Add this to your MCP client's configuration (for example `claude_desktop_config.json`, `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "pixel-sprite": {
      "command": "uvx",
      "args": ["--torch-backend=auto", "mcpbytes-pixel-sprite", "mcp"]
    }
  }
}
```

Run `mcpbytes-pixel-sprite download` once first. A 16 GB download would outlast any client's timeout, so the tool
asks for it instead of starting it. The server provides one tool, `create_pixel_sprite`, with the options above
plus `output_dir`. It writes the files, returns their summary and shows each sheet enlarged 4× so the model can
look at it. Files go to `~/mcpbytes/pixel-sprites/<timestamp>/` unless the call passes `output_dir`, or you set
`MCPBYTES_OUTPUT_DIR`. The models load on the first call and stay in memory.

## Cloud mode (optional)

`--cloud` sends the job to MCPBytes GPUs instead and charges prepaid credits (prices at
[mcpbytes.com/pricing](https://mcpbytes.com/pricing)). Create a key at
[console.mcpbytes.com/keys](https://console.mcpbytes.com/keys):

```bash
export MCPBYTES_API_KEY=...
uvx mcpbytes-pixel-sprite generate "A forest ranger" --cloud
uvx mcpbytes-pixel-sprite mcp --cloud      # the MCP server, backed by the hosted tool
```

The files are the same. The run prints a request key: if it is interrupted, run it again with
`--idempotency-key <key>` to resume the same job without a second charge. Nothing is sent to MCPBytes unless you pass `--cloud`.

## Models

Nothing is bundled. These files are downloaded from Hugging Face at pinned commits and licensed to you by their
publishers (all Apache-2.0, no account needed):

- [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) — transformer, VAE, tokenizer
- [svntax-dev/pixel_spritesheet_4walk_small_lora_v1](https://huggingface.co/svntax-dev/pixel_spritesheet_4walk_small_lora_v1) — the sprite-sheet LoRA
- [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) — the Qwen3 4B text encoder

`HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN` and `HF_HUB_OFFLINE` work as usual. See [NOTICE](https://github.com/MCPBytes/mcpbytes-pixel-sprite/blob/main/NOTICE) for attributions.

## Development

```bash
uv run --with pytest pytest
```

This repository is also the `packages/pixel-sprite` submodule of [MCPBytes/tools](https://github.com/MCPBytes/tools),
the workspace it shares with `mcpbytes-core`.

## Licence

GNU AGPL v3.0 only. The files copied into every export (`IMPORTING.md` and the Unity importer script) are MIT-0, so
you can ship them in a game under any terms. The sprites you generate are yours. See [NOTICE](https://github.com/MCPBytes/mcpbytes-pixel-sprite/blob/main/NOTICE).
