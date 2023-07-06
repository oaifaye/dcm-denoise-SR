# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    print("root_path:", root_path)
    train_pipeline(root_path)

# python -m torch.distributed.launch —nproc_per_node=4 —master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch
# python train.py -opt options/train_realesrgan_x2plus.yml --launcher none

# python train_realesrgan.py -opt options/train_realesrgan_x1denoise.yml --launcher none