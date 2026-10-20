"""FLUX.2 [klein] base 4B with the pixel-spritesheet LoRA, loaded once from the pinned Hugging Face files."""

import sys
import threading

from mcpbytes_core import hub
from mcpbytes_core.gpu import require_cuda

from .inference import request_options
from .models import COMPONENTS, DOWNLOAD_COMMAND, ENCODER, LORA, TRANSFORMER

# GPU memory modes, chosen by the card's total memory. Peaks and per-image times measured at 512x512 on an
# RTX PRO 6000 with torch 2.8: "none" keeps every model resident (16.2 GiB reserved, 2.3 s); "model" moves
# whole models on and off the GPU per stage (7.8 GiB, 7.4 s; the hosted 24 GiB cards run this way);
# "sequential" streams layers (1.0 GiB, 34 s).
OFFLOAD_MODES = ("auto", "none", "model", "sequential")


def choose_offload(total_bytes):
    return "none" if total_bytes >= 20 * 2**30 else "model" if total_bytes >= 10 * 2**30 else "sequential"


class Engine:
    def __init__(self, offload="auto"):
        torch = require_cuda()
        roots = hub.locate(COMPONENTS, DOWNLOAD_COMMAND)
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
        from safetensors.torch import load_file
        from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM

        root = roots["main"]
        # Official single-file conversion of the original base weights. The Qwen encoder has native
        # Transformers keys and tied embeddings: no custom node runtime.
        transformer = Flux2Transformer2DModel.from_single_file(str(root / TRANSFORMER), config=str(root), subfolder="transformer",
                                                               torch_dtype=torch.bfloat16, local_files_only=True)
        encoder = Qwen3ForCausalLM.from_pretrained(None, config=Qwen3Config.from_pretrained(root / "text_encoder", local_files_only=True),
                                                   state_dict=load_file(str(roots["encoder"] / ENCODER)), dtype=torch.bfloat16, local_files_only=True)
        self.pipe = Flux2KleinPipeline(transformer=transformer, text_encoder=encoder,
                                       tokenizer=Qwen2Tokenizer.from_pretrained(root / "tokenizer", local_files_only=True),
                                       vae=AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=torch.bfloat16, local_files_only=True),
                                       scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler", local_files_only=True),
                                       is_distilled=False)
        self.pipe.load_lora_weights(str(roots["lora"]), weight_name=LORA, adapter_name="character", local_files_only=True)
        self.pipe.set_adapters(["character"], adapter_weights=[1.0])
        self.pipe.set_progress_bar_config(disable=True)  # stdout belongs to the MCP protocol in server mode
        if offload == "auto":
            offload = choose_offload(torch.cuda.get_device_properties(0).total_memory)
        if offload == "model":
            self.pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.torch, self.offload, self.lock = torch, offload, threading.Lock()
        print(f"pixel sprite engine ready on {torch.cuda.get_device_name()} (offload: {offload})", file=sys.stderr, flush=True)

    def render(self, description, reference, seed):
        """One 512x512 raw sheet. `reference` is an RGB image or None."""
        with self.lock:
            try:
                with self.torch.inference_mode():
                    return self.pipe(**request_options(description, reference),
                                     generator=self.torch.Generator("cuda").manual_seed(seed)).images[0]
            except Exception:
                # Return memory to the offload hooks so the next request starts clean.
                self.pipe.maybe_free_model_hooks()
                self.torch.cuda.empty_cache()
                raise
