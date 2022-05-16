import os

rootdir = r'/home/yuzun/dataset/hcp_all/LR/test'
for image in os.listdir(rootdir):
    # print(image.split('_')[0])
    # print(image)
    # print(image[:-9])
    new_name = image[:-12]+image[-4:]
    print(new_name)
    os.rename(os.path.join(rootdir,image),
              os.path.join(rootdir,new_name))