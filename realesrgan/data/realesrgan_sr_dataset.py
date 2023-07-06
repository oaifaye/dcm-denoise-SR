import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torch.nn import functional as F
from PIL import Image


@DATASET_REGISTRY.register()
class RealESRGANSRDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANSRDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.gt_size = opt['gt_size']
        self.gt_cut_scale_range = opt['gt_cut_scale_range']
        self.noise_cut_scale_range = opt['noise_cut_scale_range']
        self.gt_folder = opt['dataroot_gt']
        self.noise_paths = []
        self.mean = opt['mean']
        self.std = opt['std']
        self.noise_radio_min = opt['noise_radio_min']
        self.noise_radio_max = opt['noise_radio_max']
        self.gray_radio = opt['gray_radio']
        self.cur_noise_index = 0

        with open(self.opt['meta_info']) as fin:
            paths = [line.strip().split(' ')[0] for line in fin]
            self.paths = [os.path.join(self.gt_folder, v) for v in paths]
        # 打乱顺序
        random.shuffle(self.paths)

        noise_dir = opt['noise_dir']
        for noise_name in os.listdir(noise_dir):
            self.noise_paths.append(os.path.join(noise_dir, noise_name))
            # 打乱顺序
        random.shuffle(self.paths)

    def __getitem__(self, index):
        t = time.time()
        if self.file_client is None:
            self.file_client = FileClient('disk')

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                t = time.time()
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=False)

        # gt随机取1-2倍，在变小
        img_gt = self.__random_cut(img_gt, self.gt_cut_scale_range)

        # gt随机旋转 翻转
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        img_gt = img_gt / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

        return_d = {'gt': img_gt, 'lq': img_gt }
        return return_d

    def __len__(self):
        return len(self.paths)

    def __random_cut(self, img, cut_scale_range):
        img_h, img_w, _ = img.shape
        scale = rand(cut_scale_range[0], cut_scale_range[1])
        w_max = img_w - self.gt_size * scale
        h_max = img_h - self.gt_size * scale
        if w_max < 1 or h_max < 1:
            scale = 1
            rand_x = 0
            rand_y = 0
        else:
            rand_x = np.random.randint(0, w_max)
            rand_y = np.random.randint(0, h_max)
        img = img[rand_y: rand_y + int(self.gt_size * scale), rand_x: rand_x + int(self.gt_size * scale), :]
        img = cv2.resize(img, (self.gt_size, self.gt_size))
        return img

    def __random_gray(self, img_gt, img_lq):
        if rand() < self.gray_radio:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.repeat(img_gt[..., np.newaxis], 3, axis=2)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.repeat(img_lq[..., np.newaxis], 3, axis=2)
        return img_gt, img_lq

    def __merge_image(self, img1, img2, top_left=(0, 0)):
        # 打开背景
        bg = Image.fromarray(img1.astype('uint8')).convert('RGB')
        # 创建底图
        target = Image.new('RGBA', (bg.size[0], bg.size[1]), (0, 0, 0, 0))
        # 打开水印
        img2_scale = img2.copy()
        # img2_scale = cv2.resize(img2_scale, (w2, h2))
        img2_scale = Image.fromarray(img2_scale.astype('uint8')).convert('RGBA')
        # 分离透明通道
        r, g, b, a = img2_scale.split()
        # 将背景贴到底图
        bg.convert("RGBA")
        target.paste(bg, (0, 0))
        # 将水印贴到底图
        img2_scale.convert("RGBA")
        startX = top_left[0]
        startY = top_left[1]
        target.paste(img2_scale, (startX, startY), mask=a)
        target = np.array(target)
        return target


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a