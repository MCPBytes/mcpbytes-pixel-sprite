"""Local and cloud runs with the model replaced: seeds, outputs, options and the hosted API exchange."""

import contextlib
import hashlib
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from mcpbytes_core.cloud import Cloud
from mcpbytes_core.errors import ToolError
from mcpbytes_pixel_sprite.engine import Engine
from mcpbytes_pixel_sprite.generate import run_cloud, run_local, validate


def render():
    sheet = Image.new("RGBA", (128, 128), (255, 0, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row in range(4):
        for col in range(4):
            draw.rectangle((col * 32 + 10, row * 32 + 8, col * 32 + 20, row * 32 + 27), fill=(30, 110, 45, 255))
    return sheet.resize((512, 512), Image.Resampling.NEAREST)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def render(self, description, reference, seed):
        self.calls.append((description, reference, seed))
        return render()


def test_local_run_uses_the_hosted_stage_seeds_and_keeps_the_prompt_out_of_generation_json(tmp_path):
    engine = FakeEngine()
    result = run_local({"description": " a ranger ", "variants": 2, "seed": 42}, out=tmp_path / "out", engine=engine, progress=lambda m: None)
    # Golden values from the Worker's derivation (sha256 of "42:sprite_N", first 32 bits), computed in JavaScript.
    assert [seed for _, _, seed in engine.calls] == [1010382725, 2330324371]
    assert all(description == "a ranger" for description, _, _ in engine.calls)
    assert result["files"] == sorted(["generation.json"] + [f"{kind}_{n}.{ext}" for n in (1, 2) for kind, ext in
                                                            (("raw", "png"), ("sprite", "png"), ("sprite", "json"), ("sprite", "zip"))])
    generation = (tmp_path / "out" / "generation.json").read_text(encoding="utf-8")
    assert "ranger" not in generation and json.loads(generation)["options"]["seed"] == 42
    assert [v["sheet"] for v in result["variants"]] == ["sprite_1.png", "sprite_2.png"]


def test_reference_is_bounded_and_recorded_by_hash(tmp_path):
    engine, source = FakeEngine(), tmp_path / "ref.png"
    Image.new("RGB", (2048, 1024), "green").save(source)
    run_local({"description": "a ranger", "seed": 1}, source, tmp_path / "out", engine=engine, progress=lambda m: None)
    assert engine.calls[0][1].size == (1024, 512)
    with zipfile.ZipFile(tmp_path / "out" / "sprite_1.zip") as archive:
        assert json.loads(archive.read("provenance.json"))["reference_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    Image.new("RGB", (8, 8)).save(tmp_path / "ref.gif")
    with pytest.raises(ToolError, match="PNG or JPEG"):
        run_local({"description": "a ranger"}, tmp_path / "ref.gif", tmp_path / "other", engine=engine)


def test_results_are_never_written_over_existing_files(tmp_path):
    (tmp_path / "sprite_1.png").write_bytes(b"mine")
    engine = FakeEngine()
    with pytest.raises(ToolError) as error:
        run_local({"description": "a ranger"}, out=tmp_path, engine=engine)
    assert error.value.code == "output_exists" and not engine.calls
    assert (tmp_path / "sprite_1.png").read_bytes() == b"mine"


@pytest.mark.parametrize("change", [
    {"description": "   "}, {"description": "x" * 4001}, {"variants": 0}, {"variants": 5}, {"variants": True},
    {"palette_colors": 24}, {"frame_duration_ms": 30}, {"frame_duration_ms": 1010}, {"frame_duration_ms": 125},
    {"palette": "#ffffff"}, {"palette": "#ffffff, #000000"}, {"palette": "#fff,#000"}, {"seed": -1}, {"seed": 2**32},
    {"prompt": "a ranger"},
])
def test_options_outside_the_hosted_rules_are_rejected(change):
    with pytest.raises(ToolError) as error:
        validate({"description": "a ranger", **change})
    assert error.value.code == "invalid_options"


def test_options_get_the_hosted_defaults():
    assert validate({"description": "a ranger", "palette": "#ffffff,#000000", "seed": 0xFFFFFFFF, "frame_duration_ms": 40}) == {
        "description": "a ranger", "variants": 1, "transparent": True, "palette_colors": 32, "frame_duration_ms": 40,
        "palette": "#ffffff,#000000", "seed": 0xFFFFFFFF}


def test_failed_inference_returns_gpu_memory_before_the_next_request():
    cleanup = []

    class Pipe:
        def __call__(self, **kwargs):
            raise RuntimeError("out of memory")

        def maybe_free_model_hooks(self):
            cleanup.append("offload")

    engine = Engine.__new__(Engine)
    engine.lock, engine.pipe = threading.Lock(), Pipe()
    engine.torch = SimpleNamespace(inference_mode=contextlib.nullcontext, Generator=lambda device: SimpleNamespace(manual_seed=lambda s: None),
                                   cuda=SimpleNamespace(empty_cache=lambda: cleanup.append("cache")))
    with pytest.raises(RuntimeError):
        engine.render("a ranger", None, 1)
    assert cleanup == ["offload", "cache"]


@pytest.fixture
def api(tmp_path):
    """A stand-in for api.mcpbytes.com that records requests and serves a finished two-candidate job."""
    published = tmp_path / "published"
    run_local({"description": "a ranger", "variants": 2}, out=published, engine=FakeEngine(), progress=lambda m: None)
    files = {path.name: path.read_bytes() for path in published.iterdir()}
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_POST(self):
            payload = json.loads(self.body())
            seen.append(("POST", self.path, dict(self.headers), payload))
            if self.path == "/v1/uploads":
                return self.reply(201, {"upload_id": "up_" + "a" * 26, "method": "PUT", "url": f"{base}/blob/up/1?t=x"})
            if payload["options"].get("palette_colors") == 64:
                return self.reply(402, {"error": {"code": "insufficient_credits", "message": "not enough credits"}})
            self.reply(202, {"id": "j_" + "b" * 26, "status": "queued"})

        def do_PUT(self):
            seen.append(("PUT", self.path, dict(self.headers), self.body()))
            self.reply(200, {})

        def do_GET(self):
            seen.append(("GET", self.path, dict(self.headers), None))
            if self.path.startswith("/files/"):
                return self.reply(200, files[self.path[7:]], "application/octet-stream")
            self.reply(200, {"id": "j_" + "b" * 26, "status": "succeeded", "credits_charged": 40, "result": {"files": [
                {"name": name, "url": f"{base}/files/{name}", "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()]}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield Cloud("test-key", base), seen, files
    server.shutdown()


def test_cloud_run_uploads_the_reference_replays_by_key_and_verifies_downloads(api, tmp_path, monkeypatch):
    cloud, seen, files = api
    monkeypatch.setattr("time.sleep", lambda s: None)
    source = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), "green").save(source)
    result = run_cloud({"description": "a ranger", "variants": 2}, source, tmp_path / "out", cloud, "my-key", progress=lambda m: None)
    upload, put, start = seen[0], seen[1], seen[2]
    assert upload[3] == {"filename": "ref.png", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size}
    assert put[3] == source.read_bytes() and "Authorization" not in put[2]  # the signed URL, never the API key
    assert start[1] == "/v1/tools/create_pixel_sprite/jobs" and start[2]["Idempotency-Key"] == "my-key"
    assert start[2]["Authorization"] == "Bearer test-key" and start[3]["upload_id"] == "up_" + "a" * 26
    assert start[3]["options"]["description"] == "a ranger" and "seed" not in start[3]["options"]
    assert result["credits_charged"] == 40 and len(result["variants"]) == 2
    assert all((tmp_path / "out" / name).read_bytes() == data for name, data in files.items())


def test_cloud_errors_carry_the_api_code(api, tmp_path):
    cloud, _, _ = api
    with pytest.raises(ToolError) as error:
        run_cloud({"description": "a ranger", "palette_colors": 64}, out=tmp_path / "out", cloud=cloud, progress=lambda m: None)
    assert error.value.code == "insufficient_credits"
