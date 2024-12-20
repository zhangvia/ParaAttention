import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.float16,
).to("cuda")

# Enable memory savings
# pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

from para_attn.cubic_attn.diffusers_adapters import cubify_pipe

cubify_pipe(pipe)

# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
video = pipe(
    prompt,
    num_frames=84,
).frames[0]

print("Saving video to mochi.mp4")
export_to_video(video, "mochi.mp4", fps=30)
