"""mcpbytes-pixel-sprite: 32x32 pixel-art character sprite sheets on your own GPU (or MCPBytes' with --cloud).

  mcpbytes-pixel-sprite download                      fetch the pinned model files (~16 GB, once)
  mcpbytes-pixel-sprite generate "a forest ranger"    write sheets, frames, walk GIFs, Godot/Unity exports
  mcpbytes-pixel-sprite mcp                           serve the create_pixel_sprite tool over MCP (stdio)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from mcpbytes_core import hub
from mcpbytes_core.errors import ToolError

from .engine import OFFLOAD_MODES
from .models import COMPONENTS


def log(message):
    print(message, file=sys.stderr, flush=True)


def download():
    if pending := hub.pending_bytes(COMPONENTS):
        log(f"Downloading {pending / 1e9:.1f} GB of pinned model files from Hugging Face into its cache (resumable)")
    hub.fetch(COMPONENTS)
    log("Model files ready")


def generate(args):
    from .generate import run_cloud, run_local

    options = {"description": args.description, "variants": args.variants, "transparent": not args.opaque,
               "palette_colors": args.palette_colors, "frame_duration_ms": args.frame_duration_ms,
               "palette": args.palette, "seed": args.seed}
    out = args.out or Path("pixel-sprites") / time.strftime("%Y%m%d-%H%M%S")
    if args.cloud:
        return run_cloud(options, args.reference, out, idempotency_key=args.idempotency_key, progress=log)
    if hub.missing(COMPONENTS):
        download()
    return run_local(options, args.reference, out, offload=args.offload, progress=log)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mcpbytes-pixel-sprite", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download the pinned model files (~16 GB) into the Hugging Face cache")
    g = commands.add_parser("generate", help="generate 1-4 candidate sprite sheets from a description")
    g.add_argument("description", help="one character, up to 4,000 characters")
    g.add_argument("--reference", type=Path, help="PNG or JPEG of the character (yours to use)")
    g.add_argument("--variants", type=int, default=1, help="candidate sheets, 1-4 (default 1)")
    g.add_argument("--opaque", action="store_true", help="keep the background instead of removing it")
    g.add_argument("--palette-colors", type=int, default=32, choices=(16, 32, 64))
    g.add_argument("--palette", help="shared palette: 2-64 comma-separated #RRGGBB colors, no spaces")
    g.add_argument("--frame-duration-ms", type=int, default=120, help="walk preview frame time, 40-1000 in steps of 10")
    g.add_argument("--seed", type=int, help="0-4294967295; the same seed, inputs and hardware repeat a result")
    g.add_argument("-o", "--out", type=Path, help="new or empty folder (default: pixel-sprites/<timestamp>)")
    g.add_argument("--cloud", action="store_true", help="run on MCPBytes GPUs for credits (needs MCPBYTES_API_KEY)")
    g.add_argument("--idempotency-key", help="with --cloud: the key printed by an interrupted run, to resume it without a new charge")
    m = commands.add_parser("mcp", help="run the MCP server on stdio (for Claude, Cursor, VS Code and other MCP clients)")
    m.add_argument("--cloud", action="store_true", help="serve the tool from MCPBytes GPUs for credits (needs MCPBYTES_API_KEY)")
    for command in (g, m):
        command.add_argument("--offload", choices=OFFLOAD_MODES, default="auto",
                             help="GPU memory mode: none needs ~17 GB, model ~8 GB, sequential less (slowest); auto picks by GPU size")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            download()
        elif args.command == "generate":
            print(json.dumps(generate(args), indent=2))
        else:
            from .server import serve

            serve(cloud=args.cloud, offload=args.offload)
    except ToolError as e:
        log(f"mcpbytes-pixel-sprite: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
