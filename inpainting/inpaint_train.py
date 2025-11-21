from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from inpainting.defect_dataset import DefectDataset as MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# Configs
#resume_path = '/home/vincent/Documents/yq/control_sd15_finetune.ckpt'
# resume_path = './models/control_sd15_ini.ckpt'
# resume_path = './lightning_logs/version_0/checkpoints/epoch=57-step=92393.ckpt'

batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
# model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))

model_name = 'control_v11p_sd15_inpaint'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
# ddim_sampler = DDIMSampler(model)



model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], accumulate_grad_batches=4)


# Train!
trainer.fit(model, dataloader)
