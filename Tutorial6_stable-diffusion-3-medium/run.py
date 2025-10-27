import torch
import torch_npu
from diffusers import StableDiffusion3Pipeline

# 检查 NPU 是否可用
device = torch.device("npu:0") if torch.npu.is_available() \
    else torch.device("cpu")

# 加载模型
pipe = StableDiffusion3Pipeline.from_pretrained(
    "./stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16
)

# 使用 NPU
pipe = pipe.to(device)

# prompt 内容，可以使用多个 prompt
# prompt2 = "Photorealistic"
prompt = ("Albert Einstein leans forward, holds a Qing dynasty fan. "
"A butterfly lands on the blooming peonies in the garden. "
"The fan is positioned above the butterfly.")

# 根据 prompt 生成多张图片
for i in range(10):
    image = pipe(
        prompt=prompt,
        # prompt_2=prompt2,
        negative_prompt=\
            "ugly, deformed, disfigured, poor details, bad anatomy",
        num_inference_steps=70,
        guidance_scale=7,
        height=1024,
        width=1024,
    ).images[0]

    image.save(f"{i}.png")