"""Shared model-family settings for preparation, training and generation."""

PROFILES = {
    "sd15": {
        "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "resolution": 512,
        "trainer": "examples/text_to_image/train_text_to_image_lora.py",
        "output": "outputs/diffusion/lora",
    },
    "sdxl": {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "resolution": 1024,
        "trainer": "examples/text_to_image/train_text_to_image_lora_sdxl.py",
        "output": "outputs/diffusion/lora-sdxl",
    },
}


def dataset_path(architecture):
    return f"data/processed/openmoji-{PROFILES[architecture]['resolution']}"
