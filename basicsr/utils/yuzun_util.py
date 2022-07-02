from __future__ import print_function
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
# from torchvision.transforms import InterpolationMode,ToTensor,Normalize
from torchvision.transforms import ToTensor,Normalize
import torch
import math
import nibabel as nib
import imageio

def fft_2D(input_Tensor,mask_ratio = 0.25):
    """使用2D傅里叶变换，将经过base_dataset，transformer,标准化后的(C,H,W)的Tensor
    转换到频域，然后使用低通滤波器滤波，面积为原图*mask_ratio
    """
    #print(input_Tensor.shape)
    img = input_Tensor.squeeze(dim=0)  # (H,W)

    # 创建一个低通滤波器
    mask = np.zeros(img.shape)
    crow = int((img.shape[0]) / 2)  # 中心位置
    ccol = int((img.shape[1]) / 2)  # 中心位置
    mask_l = int(math.sqrt(img.shape[0] * img.shape[1] * mask_ratio) / 2)
    mask[crow - mask_l:crow + mask_l, ccol - mask_l:ccol + mask_l] = 1

    f = np.fft.fft2(img)  # 2D傅里叶变换
    fshift = np.fft.fftshift(f)  # 低频中心化
    # 滤波
    fshift = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)  # 逆中心化
    iimg = np.fft.ifft2(ishift)  # 傅里叶逆变换
    iimg = np.abs(iimg)  # 去掉虚部 此时数据为float64的ndarray
    iimg = np.expand_dims(iimg, axis=-1)  # 回到了原来的函数
    #print(iimg.shape)
    # 变成Tensor
    to_tensor = ToTensor()
    iimg_Tensor = to_tensor(iimg).float()  # 从double float64转换为float 32
    # 归一化
    nor = Normalize((0.5,), (0.5,))
    iimg_Tensor = nor(iimg_Tensor)

    return iimg_Tensor


class Image_path():

    def __init__(self):
        self.result_rootpath = r'/home/yuzun/project/MRI_multiModel_SR/MRI_GAN-master/results/' # 模型所在的根目录

    def find_image(self,models_name,image_name,epochs_name=None):
        """
        找指定图片的各个模型各个epoch如果没有指定epoch_name就自己遍历把全部都找出来
        @param models_name: list,模型的名字
        @param epochs_name: list,epoch的名字，没有指定的话就遍历文件夹下的所有epoch
        @param image_name:  图片的名字，如 117324_155_real_T1.png
        @return: list,包含这些图片的绝对地址
        """
        images_path = []
        flag = False

        for model_name in models_name:
            model_path = os.path.join(self.result_rootpath,model_name)
            if epochs_name is None:
                epochs_name = []
                for epoch_name in (os.listdir(model_path)):
                    epochs_name.append(epoch_name)

            # 按照epochs_names中遍历epoch,在文件夹中寻找图片
            for item in epochs_name:
                images_dir = os.path.join(model_path,item) # 图片所在的地址
                images_dir = os.path.join(images_dir,'images')
                for image in os.listdir(images_dir):
                    if image == image_name:
                        images_path.append(os.path.join(images_dir,image))
                        flag=True
                if not flag:
                    print('{} have no matching image'.format(images_dir))
                flag = False # 将flag归为
        return images_path

    def find_imagename(self,realImage_path,root_path=None,suffix=None):
        """
        根据真实图片的绝对路径提取图片名字，加上指定后缀，指定根目录，返回经过处理的图片保存路径
        @param realImage_path: 测试的真实图片的绝对地址，如r'/home/yuzun/project/MRI_multiModel_SR/MRI_GAN-master/results/UnetD1/test_75/images/117324_155_real_T1.png'
        @return image_name：图片的名字   epoch: 图片的周期  fakeImage_path: 生成图片的地址 project_name：模型的名字
        """
        image_name = realImage_path.split('/')[10][:-12]  #'117324_155'
        epoch = realImage_path.split('/')[8]
        fakeImage_path = '/'.join(realImage_path.split('/')[:10]) + '/' + image_name + '_synthesis.png'
        project_name = realImage_path.split('/')[7]

        return image_name,epoch,fakeImage_path,project_name

    def abs_diff(self,realImage_path,fakeImage_path,show=False):
        real_img = cv2.imread(realImage_path, 0).astype(np.float64)
        fake_img = cv2.imread(fakeImage_path, 0).astype(np.float64)
        diff_img = abs(fake_img - real_img) / 255
        if show:
            print(diff_img.sum())
        return diff_img,diff_img.sum()

    def hot_map(self,diff_img,save_path):
        """
        画出热力图并保存
        @param diff_img:diff_img = abs(fake_img-real_img) / 255
        @param save_path:热力图保存的地址，绝对地址带png
        """
        hot_map = sns.heatmap(data=diff_img, vmin=0, vmax=0.15)
        figure = hot_map.get_figure()
        figure.savefig(save_path)
        plt.close('all')  # 关掉图层，以免以前的图干扰




def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data  # (batch,c,h,w)
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array  (c,h,w)
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))  # 将灰度变为RGB，在维度0复制3遍，即(3,208,208),这里的范围是(-1,0)
        image_numpy = (np.transpose(image_numpy,
                                    (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy[image_numpy < 0] = 0  # 为了抑制雪花点，但是输出不用调吗？
        # print(image_numpy)
        # 转置，(208,208,3)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)  # 输出的是一个ndarray，将float转换成uint8的时候会损失掉小数

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), InterpolationMode.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), InterpolationMode.BICUBIC)
    image_pil.save(image_path)


class ReadWrite_nii():
    def __init__(self,root_dir,nii_name,output_root_dir):
        """
        将nii文件切成图片并且保存
        @param nii_name:nii文件名
        @param root_dir:nii文件的根目录
        @param output_root_dir:输出的地址
        """
        # 定义输入输出路径，切nii的方向
        self.root_dir = root_dir
        self.nii_name = nii_name
        self.output_root_dir = output_root_dir
        self.sub_name = self.nii_name.split('/')[0]
        self.output_dir = os.path.join(self.output_root_dir, self.sub_name)
        self.oriOutput_dir = os.path.join(self.output_dir, 'ori_image')
        self.kOutput_dir = os.path.join(self.output_dir, 'k_image')
        self.SROutput_dir = os.path.join(self.output_dir, 'SR_image')


    def nii2slices(self,write=False):
        self.nii_path = os.path.join(self.root_dir, self.nii_name)
        T1_img = nib.load(self.nii_path)  # 读取nii
        self.img_affine = T1_img.affine  # 由于使用nibabel图像会旋转90度，所以读取保存的时候还得保存映射信息
        self.img_header = T1_img.header
        # 图片保存时需要的数值是[0,255]uint8的值，但是getfdate是一个flat64的数值，所以转换数据类型
        T1_array = T1_img.get_fdata()

        # T1_array.shape为(176, 240, 256, 1)

        if not os.path.exists(self.oriOutput_dir):
            os.makedirs(self.oriOutput_dir)
        else:
            print('ori dir is a;ready exits')

        if write:
            for i in range(T1_array.shape[0]):  # 去掉头尾，中间选取两张图片
                # 切除hcp数据集的最高分辨率的那个面
                T1_sliced_img = T1_array[i, :, :]  # 将z方向的第三张图弄出来，(448,29,448,1)->(448,29,1)
                print(T1_sliced_img.shape)
                #T1_sliced_img = T1_sliced_img.squeeze(-1)
                for x in np.nditer(T1_sliced_img,op_flags=['readwrite']):   # 读写模式
                    if x < 30:
                        x[...] = 0
                new_count = f"{f'{i:.{0}f}':>{0}{5}}"
                imageio.imwrite(os.path.join(self.oriOutput_dir, '{}'.format(new_count) + '.png'),
                                T1_sliced_img)

    def kspace_im(self):
        if not os.path.exists(self.kOutput_dir):
            os.makedirs(self.kOutput_dir)
        else:
            print('k dir is a;ready exits')

        # 遍历图片，并进行ksapce转换
        for ori_img in os.listdir(self.oriOutput_dir):
            ori_img_path = os.path.join(self.oriOutput_dir,ori_img)
            A = Image.open(ori_img_path).convert('RGB')
            transform_list = []
            transform_list.append(transforms.Grayscale(1))
            transform_list += [transforms.ToTensor()]
            A_transform = transforms.Compose(transform_list)
            # 这里会对图片进行normalization，通过添加定义LR,HR将图片进行下采样
            B = A_transform(A)  # 没有经过normalize，经过其他步骤的Tensor
            A = fft_2D(B)  # 经过傅里叶变换，模糊,归一化，再反傅里叶变换回来
            A = A.unsqueeze(dim=0)  # 没有bachsize这个维度，将(C,H,W)扩展成(B,C,H,W)
            # A就是经过傅里叶变换的图片,A已经是一个tensor了
            A = tensor2im(A)
            print(A.shape)
            save_image(A,os.path.join(self.kOutput_dir,ori_img))

    def slices2nii(self,SR_dir,nii_out_path):
        """
        将图片写回为nii文件
        @param SR_dir:存放图片的地址
        @param nii_out_path:nii文件存放地址，如 ESRGAN/SR.nii.gz

        """
        imgs = []
        imgs1 = []
        imgs2 = []

        # SRimg_rootdir = r'/data/yuzun/realESRGAN_result/ADNI_002_S_0413_2017'
        # SR_dir = r'/data/yuzun/SRsegment/hcp/117324/SR_image/ESRGAN_hpc4/test_latest/images'
        # SRimg_rootdir = os.path.join(SR_rootdir, 'SR_xzimage')
        # yzSRimg_rootdir = os.path.join(SR_rootdir, 'SR_yzimage')

        # for SR_img in sorted(os.listdir(SRimg_rootdir)):
        #     img = imageio.imread(os.path.join(SRimg_rootdir, SR_img))
        #     imgs.append(img)
        # xz_imgs_array = np.array(imgs, dtype=float)
        # xz_imgs_array = xz_imgs_array.transpose(1, 0, 2)  # 交换第一个和第二个维度
        #
        # for SR_img1 in sorted(os.listdir(yzSRimg_rootdir)):
        #     img1 = imageio.imread(os.path.join(yzSRimg_rootdir, SR_img1))
        #     imgs1.append(img1)
        # yz_imgs_array = np.array(imgs1, dtype=float)

        for SR_img2 in sorted(os.listdir(SR_dir)):
            img2 = imageio.imread(os.path.join(SR_dir, SR_img2))
            imgs2.append(img2)
        xy_imgs_array = np.array(imgs2, dtype=np.float32)
        #print(xy_imgs_array.shape)
        #xy_imgs_array = xy_imgs_array.transpose(1, 2, 0)  # 交换第一个和第二个维度
        new_Nife1Image = nib.Nifti1Image(xy_imgs_array, self.img_affine, self.img_header)
        # nii_output_path = r'/data/yuzun/SRsegment/hcp/117324/SR_image/ESRGAN_hpc4'
        if not os.path.exists(self.SROutput_dir):
            os.makedirs(self.SROutput_dir)
        else:
            print('SR dir is a;ready exits')

        nib.save(new_Nife1Image, os.path.join(self.SROutput_dir, nii_out_path))
        print('already save nii file!')





