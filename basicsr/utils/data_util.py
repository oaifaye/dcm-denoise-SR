# coding=utf-8
# ================================================================
#
#   File name   : data_util.py
#   Author      : Faye
#   E-mail      : xiansheng14@sina.com
#   Created date: 2023/5/25 16:38 
#   Description :
#
# ================================================================

import os


def make_meta_info_txt(img_dir, txt_path):
    txt_content = ""
    for i, img_name in enumerate(os.listdir(img_dir)):
        if os.path.isdir(os.path.join(img_dir, img_name)):
            continue
        if i != 0:
            txt_content += "\n"
        txt_content += img_name
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt_content)

