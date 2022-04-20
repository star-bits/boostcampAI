import os
from PIL import Image

print('/opt/ml/input/data/ICDAR19_LKJ/images')

for image in os.listdir("/opt/ml/input/data/ICDAR19_LKJ/images"):
    im = Image.open(os.path.join("/opt/ml/input/data/ICDAR19_LKJ/images",image))
    if im.mode != 'RGB' and im.mode != 'RGBA':
        print(image, im.mode)

print('/opt/ml/input/data/ICDAR17_LKJ/images')

for image in os.listdir("/opt/ml/input/data/ICDAR17_LKJ/images"):
    im = Image.open(os.path.join("/opt/ml/input/data/ICDAR17_LKJ/images",image))
    if im.mode != 'RGB' and im.mode != 'RGBA':
        print(image, im.mode)

# /opt/ml/input/data/ICDAR19_LKJ/images
# tr_img_01674.gif P
# tr_img_01346.png P
# tr_img_01438.gif P
# tr_img_01016.png P
# tr_img_01626.png P
# /opt/ml/input/data/ICDAR17_LKJ/images
# tr_img_1201.png P
# tr_img_1188.gif P
# tr_img_1194.png P
# tr_img_1187.gif P
# tr_img_1204.png P