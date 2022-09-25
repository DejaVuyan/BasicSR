import cv2
import os

rootdir = r'/data/yuzun/dataset/ADNI1_x/A/test'
otuput_dir = r'/data/yuzun/dataset/ADNI1_x/A/test_256'
# for image in os.listdir(rootdir):
#     # image_path = os.path.join(rootdir,image)
image_path = r'/data/yuzun/dataset/ADNI1_x/A/7_4_test/041_S_1010__110.png'
output_path =  r'/data/yuzun/dataset/ADNI1_x/A/7_4_test/041_S_1010__110_256.png'

size = (256, 256)
img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
img_new = cv2.resize(img, size, interpolation = cv2.INTER_CUBIC)   # 就是bicubic
cv2.imwrite(output_path, img_new )
print(output_path)