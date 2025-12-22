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
import os

from inpainting.infer_inpaint import InferenceInpaint
import shutil

class InferenceInpaintMultiples (InferenceInpaint):
    def __init__(self):
        super().__init__()
        return
    def init_model(self):
        model_name = 'control_v11p_sd15_inpaint'
        model = create_model(f'./models/{model_name}.yaml').cpu()
        # model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
        # model_path = f'./models/{model_name}.pth'
        # model_path = './lightning_logs/version_7/checkpoints/epoch=110-step=16649.ckpt'
        model_path = './lightning_logs/version_9/checkpoints/epoch=99-step=4099.ckpt'
        # model_path = './lightning_logs/version_10/checkpoints/epochepoch=199-stepstep=1999.ckpt'
        model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)

        self.model = model
        self.ddim_sampler = ddim_sampler
        return
    def run_one_infer(self, idx, prompt, mask, image_path):
        a_prompt = ''
        # n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
        n_prompt = ''
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


        input_image_and_mask['image'] = imageio.imread(image_path) 
        if input_image_and_mask['image'].ndim == 2:
            input_image_and_mask['image'] = np.stack([input_image_and_mask['image']] * 3, axis=-1)
        elif input_image_and_mask['image'].shape[2] == 4:
            input_image_and_mask['image'] = input_image_and_mask['image'][..., :3]
        input_image_and_mask['mask'] = np.zeros_like(input_image_and_mask['image'], dtype=np.uint8) 
        input_image_and_mask['mask'][mask[0]:mask[1], mask[2]:mask[3], :] = 255

        input_image_and_mask['orig_image'] = input_image_and_mask['image'].copy()
        # input_image_and_mask['image'][mask[0]:mask[1], mask[2]:mask[3], :] = -255
        
        
        res = self.process(input_image_and_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur)
        input_image, input_mask, mask_pixel, detected_map, output_images = res 
        # Save output images to disk
        infer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp/infer/multiples')
        os.makedirs(infer_dir, exist_ok=True)
        ref_shape = input_image_and_mask['image'].shape[:2]  # (height, width)
        img = output_images[0]
        img = cv2.resize(img, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_LINEAR)
        img_path = os.path.join(infer_dir, f'{idx}.png')
        cv2.imwrite(img_path, img)

        
        return img_path
    def get_bottom_prompt(self):
        view_angle = 'bottom'
        img_name = '20250303_163548_9bf8425a-dc21-4449-8b8f-546f277613d7.jpg'
        image_path = f"./data/images/{view_angle}/{img_name}"
        # label_path = f"./data/labels/{view_angle}/{img_name.replace('.jpg', '.txt')}"

        mask_prompts = []
        # mask_prompts.append([[470, 586, 247, 389], "Guiderod malposed, bottom viewpoint"])
        # mask_prompts.append([[474, 603, 1249, 1412], "Guiderod oil, bottom viewpoint"])

        masks = []
        prompts = []

        prompts.append("Actuator breakage, bottom viewpoint")
        masks.append([375,549,153,242])

        prompts.append("Guiderod malposed, bottom viewpoint")
        masks.append([465,585,244,397])

        prompts.append("Guiderod oil, bottom viewpoint")
        masks.append([485,606,1254,1418])

        prompts.append("Piston breakage, bottom viewpoint")
        masks.append([693,721,678,965])

        for mask, prompt in zip(masks, prompts):
            mask_prompts.append((mask, prompt))
        return image_path, mask_prompts
    def run(self):
        self.init_model()

        image_path, mask_prompts = self.get_bottom_prompt()

        infer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp/infer/multiples')
        orig_img_path = os.path.join(infer_dir, 'orig.jpg')
        shutil.copy(image_path, orig_img_path)

        # Draw all masks on the original image and save as mask.jpg
        orig_img = cv2.imread(image_path)
        mask_img = orig_img.copy()
        for mask, prompt in mask_prompts:
            y1, y2, x1, x2 = mask  # Correct order: y1, y2, x1, x2
            # Draw rectangle for mask (BGR: red)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Draw prompt text above the rectangle
            text_pos = (x1, max(y1 - 10, 0))
            cv2.putText(mask_img, prompt, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        mask_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp/infer/multiples/mask.jpg')
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        cv2.imwrite(mask_save_path, mask_img)
        print(f"Mask image with all masks saved to: {mask_save_path}")

        for idx, (mask, prompt) in enumerate(mask_prompts):
            image_path = self.run_one_infer(idx, prompt, mask, image_path)
            print(f"Result saved to: {image_path}")
        return
    

if __name__ == "__main__":
    infer_inpaint = InferenceInpaintMultiples()
    infer_inpaint.run()
