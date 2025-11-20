from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import imageio
import matplotlib.pyplot as plt
import numpy as np



class InferenceInpaint:
    def __init__(self):
        
        return
    def init_model(self):
        model_name = 'control_v11p_sd15_inpaint'
        model = create_model(f'./models/{model_name}.yaml').cpu()
        model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
        model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)

        self.model = model
        self.ddim_sampler = ddim_sampler
        return
    def run(self):
        prompt = "oil"
        a_prompt = 'best quality, extremely detailed'
        n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
        num_samples = 1
        image_resolution = 512
        ddim_steps = 20
        guess_mode = False
        strength = 1.0
        scale = 9.0
        seed = -1
        eta = 1.0
        mask_blur = 5.0

        input_image_and_mask = {}
        input_image_and_mask['image'] = None 
        input_image_and_mask['mask'] = None

        img_name = '20250303_163624_4c3b29c3-254b-4f6c-9e13-a5ab6c85f377.jpg'
        image_path = f"./data/images/bottom/{img_name}"
        mask = [470,586,247,389]

        input_image_and_mask['image'] = imageio.imread(image_path) 
        if input_image_and_mask['image'].ndim == 2:
            input_image_and_mask['image'] = np.stack([input_image_and_mask['image']] * 3, axis=-1)
        elif input_image_and_mask['image'].shape[2] == 4:
            input_image_and_mask['image'] = input_image_and_mask['image'][..., :3]
        input_image_and_mask['mask'] = np.zeros_like(input_image_and_mask['image'], dtype=np.uint8) 
        input_image_and_mask['mask'][mask[0]:mask[1], mask[2]:mask[3], :] = 255
        
        self.init_model()
        res = self.process(input_image_and_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur)
        input_image, input_mask, mask_pixel, detected_map, output_images = res 
        self.show_pipe_result(
            input_image,
            input_mask,
            mask_pixel,
            detected_map,
            output_images,
            figsize=(15, 4),        cmap_detected='gray',  
            titles=None
        )

        return
    def show_pipe_result(self,
        input_image,          # (H, W, 3)   or  (H, W)
        input_mask,           # (H, W)      mask for the input
        mask_pixel,           # (H, W, 3)   or  (H, W)
        detected_map,         # (H, W)      e.g. canny / depth / pose …
        output_images,        # list/array  length = num_samples  (each (H, W, 3) or (H, W))
        figsize=(15, 4),
        cmap_detected='gray',
        titles=None
        ):
        """
        Display the whole pipeline in one figure.
        `output_images` can be a single image or a list / NumPy array of images
        (length = num_samples).  All images are shown in a single row.
        """
        # make sure we have a list of outputs
        num_samples = len(output_images)
        total_cols = 4 + num_samples          # input_image | input_mask | mask_pixel | detected_map | outputs …
        fig, axes = plt.subplots(1, total_cols, figsize=figsize)
        if total_cols == 1:                   # plt.subplots behaviour when ncols=1
            axes = [axes]

        def _imshow(ax, img, title, cmap=None):
            if len(img.shape) == 2:           # 2-D → grayscale
                ax.imshow(img, cmap=cmap or 'gray')
            else:                             # 3-D
                ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        # default titles
        if titles is None:
            titles = ['input_image', 'input_mask', 'mask_pixel', 'detected_map'] + \
                    [f'output_{i}' for i in range(num_samples)]


        # first four fixed images
        _imshow(axes[0], input_image, titles[0])
        _imshow(axes[1], input_mask,  titles[1])
        _imshow(axes[2], mask_pixel,  titles[2])
        _imshow(axes[3], detected_map, titles[3], cmap=cmap_detected)

        # variable number of output images
        for idx, out_img in enumerate(output_images):
            _imshow(axes[4 + idx], out_img, titles[4 + idx])

        plt.tight_layout()
        plt.show()

    
    def process(self, input_image_and_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur):
        model = self.model
        ddim_sampler = self.ddim_sampler
        with torch.no_grad():
            input_image = HWC3(input_image_and_mask['image'])
            input_mask = input_image_and_mask['mask']

            img_raw = resize_image(input_image, image_resolution).astype(np.float32)
            H, W, C = img_raw.shape

            mask_pixel = cv2.resize(input_mask[:, :, 0], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)

            mask_latent = cv2.resize(mask_pixel, (W // 8, H // 8), interpolation=cv2.INTER_AREA)

            detected_map = img_raw.copy()
            detected_map[mask_pixel > 0.5] = - 255.0

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            mask = 1.0 - torch.from_numpy(mask_latent.copy()).float().cuda()
            mask = torch.stack([mask for _ in range(num_samples)], dim=0)
            mask = einops.rearrange(mask, 'b h w -> b 1 h w').clone()

            x0 = torch.from_numpy(img_raw.copy()).float().cuda() / 127.0 - 1.0
            x0 = torch.stack([x0 for _ in range(num_samples)], dim=0)
            x0 = einops.rearrange(x0, 'b h w c -> b c h w').clone()

            mask_pixel_batched = mask_pixel[None, :, :, None]
            img_pixel_batched = img_raw.copy()[None]

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
            x0 = model.get_first_stage_encoding(model.encode_first_stage(x0))

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond, x0=x0, mask=mask)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(np.float32)
            x_samples = x_samples * mask_pixel_batched + img_pixel_batched * (1.0 - mask_pixel_batched)

            results = [x_samples[i].clip(0, 255).astype(np.uint8) for i in range(num_samples)]
            return [input_image, input_mask, mask_pixel] + [detected_map.clip(0, 255).astype(np.uint8)] + [results]

if __name__ == "__main__":
    infer_inpaint = InferenceInpaint()
    infer_inpaint.run()
