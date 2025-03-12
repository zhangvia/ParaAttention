import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from diffusers.utils import export_to_video

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

mesh = init_context_parallel_mesh(pipe.device.type)
parallelize_pipe(pipe, mesh=mesh)

torch._inductor.config.reorder_for_compute_comm_overlap = True
torch.set_float32_matmul_precision("high")
# pipe.transformer = torch.compile(pipe.transformer)

output = pipe(
    prompt="A cat is doing an acrobatic dive into a swimming pool at the olympics",
    negative_prompt="",
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=30,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).frames[0]

if dist.get_rank() == 0:
    export_to_video(output, "wan.mp4", fps=16)

dist.destroy_process_group()
