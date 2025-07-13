import runpod

import os
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
from mvadapter.utils import make_image_grid

def handler(event):
    print(f"Worker Start")
    input = event['input']
    
    args = {
      "device": input.get("device", "cuda"),
      "variant": input.get("variant", "sd21"),
      "mesh": input["mesh"],
      "image": input["image"],
      "text": input.get("text", "high quality"),
      "seed": input.get("seed", -1),
      "save_dir": input.get("save_dir", "./output"),
      "save_name": input.get("save_name", "i2tex_sample"),
      "reference_conditioning_scale": input.get("reference_conditioning_scale", 1.0),
      "preprocess_mesh": input.get("preprocess_mesh", False),
      "remove_bg": input.get("remove_bg", False),
    }

    if args["variant"] == "sdxl":
        from .inference_ig2mv_sdxl import prepare_pipeline, remove_bg, run_pipeline

        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_model = "madebyollin/sdxl-vae-fp16-fix"
        height = width = 768
        uv_size = 4096
    elif args["variant"] == "sd21":
        from .inference_ig2mv_sd import prepare_pipeline, remove_bg, run_pipeline

        base_model = "stabilityai/stable-diffusion-2-1-base"
        vae_model = None
        height = width = 512
        uv_size = 2048
    else:
        raise ValueError(f"Invalid variant: {args['variant']}")

    device = args["device"]
    num_views = 6

    # Prepare pipelines
    pipe = prepare_pipeline(
        base_model=base_model,
        vae_model=vae_model,
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler=None,
        num_views=num_views,
        device=device,
        dtype=torch.float16,
    )
    if args["remove_bg"]:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(args["device"])
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, args["device"])
    else:
        remove_bg_fn = None

    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="./checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="./checkpoints/big-lama.pt",
        device=device,
    )
    print("Pipeline ready.")

    os.makedirs(args["save_dir"], exist_ok=True)

    # 1. run MV-Adapter to generate multi-view images
    images, _, _, _ = run_pipeline(
        pipe,
        mesh_path=args["mesh"],
        num_views=num_views,
        text=args["text"],
        image=args["image"],
        height=height,
        width=width,
        num_inference_steps=50,
        guidance_scale=3.0,
        seed=args["seed"],
        reference_conditioning_scale=args["reference_conditioning_scale"],
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        device=device,
        remove_bg_fn=remove_bg_fn,
    )
    mv_path = os.path.join(args["save_dir"], f"{args['save_name']}.png")
    make_image_grid(images, rows=1).save(mv_path)

    torch.cuda.empty_cache()

    # 2. un-project and complete texture
    out = texture_pipe(
        mesh_path=args["mesh"],
        save_dir=args["save_dir"],
        save_name=args["save_name"],
        uv_unwarp=True,
        preprocess_mesh=args["preprocess_mesh"],
        uv_size=uv_size,
        rgb_path=mv_path,
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
    )
    print(f"Output saved to {out.shaded_model_save_path}")

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({ 'handler': handler })