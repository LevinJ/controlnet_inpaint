import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        # Stitch images into one final image with two columns
        keys = list(images.keys())
        grids = []
        max_h, max_w = 0, 0
        # First, compute the max height and width
        for k in keys:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            h, w = grid.shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            grids.append(grid)
        # Resize all grids to max_h, max_w
        resized_grids = []
        for grid in grids:
            from PIL import Image
            if grid.ndim == 2:
                grid = np.expand_dims(grid, -1)
            pil_grid = Image.fromarray(grid)
            pil_grid = pil_grid.resize((max_w, max_h), Image.BILINEAR)
            grid = np.array(pil_grid)
            resized_grids.append(grid)
        c = resized_grids[0].shape[2] if resized_grids[0].ndim == 3 else 1
        # Layout: two columns
        n = len(resized_grids)
        ncols = 2
        nrows = (n + 1) // 2
        stitched = np.zeros((max_h * nrows, max_w * ncols, c), dtype=np.uint8)
        for idx, (grid, k) in enumerate(zip(resized_grids, keys)):
            row = idx // ncols
            col = idx % ncols
            y1 = row * max_h
            y2 = y1 + max_h
            x1 = col * max_w
            x2 = x1 + max_w
            stitched[y1:y2, x1:x2, :] = grid if grid.ndim == 3 else np.expand_dims(grid, -1)
            # Draw k info on each section
            from PIL import ImageDraw, ImageFont
            pil_section = Image.fromarray(stitched[y1:y2, x1:x2, :])
            draw = ImageDraw.Draw(pil_section)
            try:
                font = ImageFont.truetype("arial.ttf", 32)
            except:
                font = ImageFont.load_default()
            draw.text((10, 10), k, fill=(255, 0, 0), font=font)
            stitched[y1:y2, x1:x2, :] = np.array(pil_section)
        filename = "stitched_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(stitched).save(path)
        return

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
