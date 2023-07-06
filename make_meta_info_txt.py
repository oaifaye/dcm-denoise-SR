# coding=utf-8
# ================================================================
#
#   File name   : make_meta_info_txt.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2023/5/25 16:45 
#   Description :
#
# ================================================================

import basicsr.utils.data_util as data_util


if __name__ == '__main__':
    data_util.make_meta_info_txt(r'datasets/DIV2K_train_HR', r'datasets/DIV2K_train_HR/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt')