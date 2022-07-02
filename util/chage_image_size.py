import cv2
import os

rootdir = r'/home/yuzun/dataset/ADNI1_x/A/test'
otuput_dir = r'/home/yuzun/dataset/ADNI1_x/A/test_320'
for image in os.listdir(rootdir):
    image_path = os.path.join(rootdir,image)
    output_path = os.path.join(otuput_dir,image)

    size = (320, 320)
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img_new = cv2.resize(img, size, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(output_path, img_new )
    print(output_path)