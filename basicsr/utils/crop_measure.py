import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image
import yuzun_util

# hcp
crop_hcp_ROI1 = [40, 40, 140, 140]
crop_hcp_ROI2 = [122, 15, 222, 115]
crop_hcp_ROI3 = [165, 175, 265, 275]
crop_hcp_ROI4 = [0, 175, 100, 275]
crop_hcp_ROIs = [crop_hcp_ROI1, crop_hcp_ROI2, crop_hcp_ROI3, crop_hcp_ROI4]


# 读入指定文件夹的图片，并且按照ROI进行crop,ROI要分ADNI1,ADNI3,HCP
# 并保存crop后的图片到另一个文件夹
def image_cut_save(path, crop_ROI,save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    left, upper, right, lower = crop_ROI
    img = Image.open(path)  # 打开图像
    box = (left, upper, right, lower)
    roi = img.crop(box)

    # 提取文件名，并且在后面加上cropi


    # 保存截取的图片
    roi.save(save_path)

# 遍历文件夹中的文件，并且提取出图片名
def walk_dir(root_dir):
    """
    遍历文件夹中的文件，并且提取出图片名,带后缀
    @param root_dir: 文件夹root_dir
    @return: image_name： 带后缀的图片文件名
    """
    image_paths = []
    image_names = []
    for image_path in os.listdir(root_dir):
        image_paths.append(image_path)
        image_name = image_path.split('/')[-1]
        print(image_name)
        image_names.append(image_name)
    return image_paths,image_names



# 计算 PSNR,SSIM
# 输出一个list，带着图片名字，能输出到CSV文件中，计算平均值和中位数标准差


# 计算NIQE，设计核的尺寸问题

#######################################
if __name__ == '__main__':
    im_paths,im_names = walk_dir(r'/home/yuzun/project/MRI_multiModel_SR/BasicSR/experiments/005_ESRGAN_hcp_80000_B16G1_wandb/visualization/826353_80')
    # for im_path in im_paths:
