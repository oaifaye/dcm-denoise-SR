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
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)


@DATASET_REGISTRY.register()
class RealESRGANDenoiseDataset(data.Dataset):
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
        super(RealESRGANDenoiseDataset, self).__init__()
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
        # self.scale = opt['scale']
        self.cost = {'t1':0,'t2':0,'t3':0,'t4':0,'t5':0,}

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
                self.cost['t1'] += time.time() - t
                t = time.time()
                img_bytes = self.file_client.get(gt_path, 'gt')
                self.cost['t2'] += time.time() - t
            except (IOError, OSError) as e:
                t = time.time()
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
                self.cost['t3'] += time.time() - t
            else:
                break
            finally:
                retry -= 1
        t = time.time()
        img_gt = imfrombytes(img_bytes, float32=False)
        # cv2.imwrite(r'C:\Users\Administrator\Desktop\zawu\11/11.jpg', img_gt)
        self.cost['t4'] += time.time() - t

        # gt随机取1-2倍，在变小
        img_gt = self.__random_cut(img_gt, self.gt_cut_scale_range)

        # gt随机旋转 翻转
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        t = time.time()
        # 随机选noise文件
        img_lq = img_gt.copy()
        noise_radio = 0.
        for i in range(10):
            # noise_path = self.noise_paths[np.random.randint(0, len(self.noise_paths))]
            # 取noise图片
            noise_path = self.noise_paths[self.cur_noise_index]
            if self.cur_noise_index == len(self.noise_paths) - 1:
                self.cur_noise_index = 0
            else:
                self.cur_noise_index += 1
            noise_bytes = self.file_client.get(noise_path, 'gt')
            img_noise = imfrombytes(noise_bytes, flag='unchanged', float32=False)

            img_noise = self.__random_cut(img_noise, self.noise_cut_scale_range)

            # img_noise_h, img_noise_w, _ = img_noise.shape
            # # 对noise图片随机截取
            # noise_cut_scale = rand(self.noise_cut_scale_range[0], self.noise_cut_scale_range[1])
            # rand_x = 0 if img_noise_w - noise_cut_scale * self.gt_size < 1 else np.random.randint(0, img_noise_w - noise_cut_scale * self.gt_size)
            # rand_y = 0 if img_noise_h - noise_cut_scale * self.gt_size < 1 else np.random.randint(0, img_noise_h - noise_cut_scale * self.gt_size)
            # img_noise = img_noise[rand_y: rand_y + int(noise_cut_scale * self.gt_size), rand_x: rand_x + int(noise_cut_scale * self.gt_size), ...]
            # img_noise = cv2.resize(img_noise, (self.gt_size, self.gt_size))

            img_noise = augment(img_noise, self.opt['use_hflip'], self.opt['use_rot'])


            cur_noise_radio = np.sum(img_noise[..., 3]) / (self.gt_size * self.gt_size * 255)
            # 如果noise太大 continue
            if cur_noise_radio + noise_radio >= self.noise_radio_max:
                continue
            # 20%概率噪声变成白色
            # 临时添加 如果噪声面积小随机变成白色
            # if noise_path.find("_") > -1 and rand() < 0.2:
            if cur_noise_radio <= 0.015 and rand() < 0.4:
                img_noise[..., 0] = int(rand(220, 255)) - img_noise[..., 0]
                img_noise[..., 1] = int(rand(220, 255)) - img_noise[..., 1]
                img_noise[..., 2] = int(rand(220, 255)) - img_noise[..., 2]
            img_lq = self.__merge_image(img_lq, img_noise)
            noise_radio += cur_noise_radio
            if noise_radio >= self.noise_radio_min:
                break

        self.cost['t5'] += time.time() - t

        img_lq = img_lq[..., [0, 1, 2]]

        # 随机变灰度图
        img_gt, img_lq = self.__random_gray(img_gt, img_lq)

        img_gt = img_gt / 255.
        img_lq = img_lq / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]

        # normalize(img_gt, self.mean, self.std, inplace=True)
        # normalize(img_lq, self.mean, self.std, inplace=True)

        # if self.scale != 1:
        #     interpolate_mode = random.choice(['area', 'bilinear', 'bicubic'])
        #     img_lq = F.interpolate(img_lq, size=(self.gt_size // self.scale, self.gt_size // self.scale), mode=interpolate_mode)

        return_d = {'gt': img_gt, 'lq': img_lq, "noise_radio": str(i) +"-"+str(noise_radio)[0:6]+'.png', "img_noise": img_noise, "cost:":self.cost }
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