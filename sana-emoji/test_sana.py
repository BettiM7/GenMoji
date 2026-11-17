import time
import torch
from diffusers import SanaPipeline

model_id = "Efficient-Large-Model/Sana_600M_512px_diffusers"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

pipe = SanaPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
)

# safer on Apple Silicon
pipe.text_encoder.to(dtype=torch.float32)
pipe.vae.to(dtype=torch.float32)

pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=False)

prompt = (
    "A sleepy penguin drinking coffee, flat emoji style, be careful with the details, body proportions, and facial expressions,"
    "simple shapes, clean outlines, white background"
)

start = time.time()

image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=4.5,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

elapsed = time.time() - start
image.save("sana_test.png")

print(f"Done. Saved sana_test.png in {elapsed:.2f} seconds")