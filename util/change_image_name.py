import os

rootdir = r'/data/yuzun/MRI_data/hcp/hcp_kspace/images/UnetD7/val/LR_images'
for image in os.listdir(rootdir):
    # print(image.split('_')[0])
    new_name = image[:-7]+image[-4:]
    print(new_name)
    os.rename(os.path.join(rootdir,image),
              os.path.join(rootdir,new_name))