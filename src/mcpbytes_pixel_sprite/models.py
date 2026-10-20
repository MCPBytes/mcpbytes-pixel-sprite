"""The exact Hugging Face files the hosted tool was qualified with (all Apache-2.0, ungated): about 16 GB.

Revisions are commits, never branches. Sizes and SHA-256s were read from the Hub for these commits.
"""

COMPONENTS = {
    "main": {
        "repository": "black-forest-labs/FLUX.2-klein-base-4B",
        "revision": "a3b4f4849157f664bdbc776fd7453c2783562f4d",
        "files": {
            "flux-2-klein-base-4b.safetensors": 7751105712,
            "vae/config.json": 821,
            "vae/diffusion_pytorch_model.safetensors": 168120878,
            "LICENSE.md": 9584,
            "README.md": 10084,
            "model_index.json": 422,
            "transformer/config.json": 531,
            "text_encoder/config.json": 1536,
            "text_encoder/generation_config.json": 214,
            "scheduler/scheduler_config.json": 486,
            "tokenizer/added_tokens.json": 707,
            "tokenizer/chat_template.jinja": 4168,
            "tokenizer/merges.txt": 1671853,
            "tokenizer/special_tokens_map.json": 613,
            "tokenizer/tokenizer.json": 11422654,
            "tokenizer/tokenizer_config.json": 5404,
            "tokenizer/vocab.json": 2776833,
        },
    },
    "lora": {
        "repository": "svntax-dev/pixel_spritesheet_4walk_small_lora_v1",
        "revision": "cb8bffe5b9b92e656e31e7f940ec4a38047d6a88",
        "files": {"pixel_4walk_small_flux2_klein_base_4b_v1.safetensors": 92426736, "README.md": 6181},
    },
    "encoder": {
        "repository": "Comfy-Org/z_image_turbo",
        "revision": "6fc90a3b1b653e935a0d175e260736de25b84df5",
        "files": {"split_files/text_encoders/qwen_3_4b.safetensors": 8044982048, "README.md": 4785},
    },
}

TRANSFORMER = "flux-2-klein-base-4b.safetensors"
LORA = "pixel_4walk_small_flux2_klein_base_4b_v1.safetensors"
ENCODER = "split_files/text_encoders/qwen_3_4b.safetensors"

# Weight hashes, recorded in each export's provenance.json as the hosted tool does.
MODEL_HASHES = sorted([
    "9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92",  # main transformer
    "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",  # main VAE
    "a968c042c2443bf6bfed41ce833a0e06bf5bbb7961ac40b88fb7257a940ae54e",  # LoRA
    "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",  # Qwen3 encoder
])

DOWNLOAD_COMMAND = "mcpbytes-pixel-sprite download"
