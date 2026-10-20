"""MCP server (stdio) exposing create_pixel_sprite, run on this machine's GPU or, with --cloud, on MCPBytes.

stdout carries the protocol: everything else goes to stderr. The models load on the first call and stay loaded.
"""

import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Literal

import anyio.from_thread
import anyio.to_thread
from pydantic import Field

from mcp.server import MCPServer
from mcp.server.mcpserver import Context, Image
from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcpbytes_core import hub
from mcpbytes_core.errors import ToolError
from PIL import Image as PILImage

from .generate import TOOL, run_cloud, run_local
from .models import COMPONENTS, DOWNLOAD_COMMAND

DESCRIPTION = (
    "Generate 1-4 pixel-art character sprite sheets from a description and an optional reference image. "
    "Each 128x128 PNG has sixteen 32x32 cells: rows face down, left, right, up; columns 0-2 are walk poses, "
    "column 3 holds arms raised, jump left, jump right and lying still. Writes raw_N.png, sprite_N.png, "
    "sprite_N.json (palette, frames, animations, QA) and sprite_N.zip (frames, walk GIFs, Godot scene, Unity importer) "
    "per candidate, plus generation.json, into output_dir. Returns the sheets enlarged 4x for review. "
    "Outputs need visual review; qa_status is never an approval. Takes seconds to under a minute per candidate, "
    "depending on GPU memory, plus model loading on the first call."
)


def default_output():
    root = Path(os.environ.get("MCPBYTES_OUTPUT_DIR") or Path.home() / "mcpbytes" / "pixel-sprites")
    return root / time.strftime("%Y%m%d-%H%M%S")


def preview(path):
    """The 128 px sheet enlarged 4x with nearest neighbour, so a model can see its pixels."""
    with PILImage.open(path) as sheet:
        target = io.BytesIO()
        sheet.resize((512, 512), PILImage.Resampling.NEAREST).save(target, format="PNG")
    return Image(data=target.getvalue(), format="png")


def build(cloud=False, offload="auto"):
    server = MCPServer("mcpbytes-pixel-sprite", instructions=(
        "create_pixel_sprite writes sprite sheets to a local folder. Pass an absolute output_dir inside the user's "
        "project when they want the files there. Look at the returned sheets before calling the result usable."))
    engine, loading = [None], threading.Lock()

    def local_engine():
        with loading:
            if engine[0] is None:
                from .engine import Engine

                engine[0] = Engine(offload)
        return engine[0]

    @server.tool(name=TOOL, title="Create a pixel sprite", description=DESCRIPTION, structured_output=False)
    async def create_pixel_sprite(
        description: Annotated[str, Field(min_length=1, max_length=4000, description="One character; a reference image should be yours to use")],
        ctx: Context,
        reference: Annotated[str | None, Field(description="Absolute path of a PNG or JPEG of the character")] = None,
        variants: Annotated[int, Field(ge=1, le=4, description="Candidate sheets")] = 1,
        transparent: Annotated[bool, Field(description="Attempt background removal; inspect QA warnings and raw output")] = True,
        palette_colors: Annotated[Literal[16, 32, 64], Field(description="Shared colors in the palette")] = 32,
        palette: Annotated[str | None, Field(description="Shared palette: 2-64 comma-separated #RRGGBB colors, no spaces")] = None,
        frame_duration_ms: Annotated[int, Field(ge=40, le=1000, multiple_of=10, description="Walk preview frame time")] = 120,
        seed: Annotated[int | None, Field(ge=0, le=0xFFFFFFFF, description="Repeat a result with the same inputs and hardware")] = None,
        output_dir: Annotated[str | None, Field(description="New or empty folder; default ~/mcpbytes/pixel-sprites/<timestamp>")] = None,
        idempotency_key: Annotated[str | None, Field(max_length=100, description="Cloud mode: reuse after a lost response to avoid a second charge")] = None,
    ) -> list:
        options = {"description": description, "variants": variants, "transparent": transparent, "palette_colors": palette_colors,
                   "palette": palette, "frame_duration_ms": frame_duration_ms, "seed": seed}
        out = Path(output_dir) if output_dir else default_output()
        steps = [0]

        def progress(message):
            steps[0] += 1
            anyio.from_thread.run(ctx.report_progress, steps[0], None, message)

        def work():
            if cloud:
                return run_cloud(options, reference, out, idempotency_key=idempotency_key, progress=progress)
            if hub.missing(COMPONENTS):
                # A 16 GB download inside a tool call would outlive any client's timeout.
                raise ToolError("models_missing", "the model files are not downloaded yet", f"ask the user to run `{DOWNLOAD_COMMAND}` once")
            if engine[0] is None:
                progress("Loading models")
            return run_local(options, reference, out, engine=local_engine(), progress=progress)

        try:
            result = await anyio.to_thread.run_sync(work)
        except ToolError as e:
            raise McpToolError(str(e)) from None
        sheets = [preview(Path(result["output_dir"]) / v["sheet"]) for v in result["variants"]]
        return [json.dumps(result, indent=2), *sheets]

    return server


def serve(cloud=False, offload="auto"):
    build(cloud, offload).run("stdio")
