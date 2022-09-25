import os

rootdir = r'/data2/yuzun/StyTR_kspace/train'
for image in os.listdir(rootdir):
    # print(image.split('_')[0])
    # print(image)
    # print(image[:-9])
    new_name = image[:-15]+image[-4:]
    print(new_name)
#     os.rename(os.path.join(rootdir,image),
#               os.path.join(rootdir,new_name))

# 将png后缀改为jpg
# for image in os.listdir(rootdir):
#     # print(image.split('_')[0])
#     # print(image)
#     # print(image[:-9])
#     new_name = image[:-4]+'.jpg'
#     if image[-4:] == '.png':
#         print(new_name)
#         os.rename(os.path.join(rootdir,image),
#               os.path.join(rootdir,new_name))