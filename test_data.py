# coding=utf-8
# ================================================================
#
#   File name   : test_data.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2023/6/5 10:54 
#   Description :
#
# ================================================================
import yaml
from realesrgan.data.realesrgan_denoise_dataset import RealESRGANDenoiseDataset
import numpy as np
import cv2
import os
from basicsr.utils import FileClient, get_root_logger, imfrombytes, tensor2img


def test_denoise_dataset():
    tmp_dir= r'C:\Users\Administrator\Desktop\zawu\22'
    # with open('options/train_realesrgan_x1denoise.yml', mode='r', encoding='utf-8') as f:
    # with open('options/train_realesrnet_x1denoise.yml', mode='r', encoding='utf-8') as f:
    with open('options/train_realesrnet_x2denoise.yml', mode='r', encoding='utf-8') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
        opt = opt['datasets']['train']

    dataset = RealESRGANDenoiseDataset(opt)
    # assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    # assert len(dataset) == 1  # whether to read correct meta info
    # assert dataset.kernel_list == ['iso', 'aniso']  # correct initialization the degradation configurations
    # assert dataset.color_jitter_prob == 1

    for i in range(500):
        result = dataset.__getitem__(i)

        lq = result['lq']
        lq = tensor2img(lq)
        # lq = lq.detach().numpy()
        # lq = np.swapaxes(lq, 2, 0) * 255
        # lq = np.asarray(lq, np.uint8)
        # # lq = lq[..., [2, 1, 0]]
        # lq = np.asarray(lq, dtype=np.uint8)
        target_path = os.path.join(tmp_dir, str(i)+'_lq.jpg')
        cv2.imwrite(target_path, lq)

        gt = result['gt']
        gt = tensor2img(gt)
        # gt = gt.detach().numpy()
        # gt = np.swapaxes(gt, 2, 0) * 255
        # gt = np.asarray(gt, np.uint8)
        # # gt = gt[..., [2, 1, 0]]
        # gt = np.asarray(gt, dtype=np.uint8)
        target_path = os.path.join(tmp_dir, str(i) + '_gt.jpg')
        cv2.imwrite(target_path, gt)

        gt = result['img_noise']
        noisr_redis = result['noise_radio']
        # gt = gt.detach().numpy()
        # gt = (np.swapaxes(gt, 2, 0) + 1) / 2 * 255
        # gt = np.asarray(gt, np.uint8)
        # gt = gt[..., [2, 1, 0]]
        # gt = np.asarray(gt, dtype=np.uint8)
        target_path = os.path.join(tmp_dir, str(i) + '_noise_'+noisr_redis+'.png')
        cv2.imwrite(target_path, gt)



if __name__ == '__main__':
    test_denoise_dataset()