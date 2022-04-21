import cv2
import os
import shutil
from tqdm import tqdm
import json
import numpy as np

data_dir = '/opt/ml/input/data/'

# dataset_dirs = ['upstage_dataset']
dataset_dirs = ['ICDAR17_LKJ', 'ICDAR19_LKJ']

long_side = 1024

for ds_dir in dataset_dirs:
    root_dir = os.path.join(data_dir, ds_dir)
    dst_path = os.path.join(data_dir, ds_dir + '_resize')
    os.makedirs(os.path.join(dst_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'ufo'), exist_ok=True)

    # with open(os.path.join(root_dir, 'ufo/annotation.json'), 'r') as f:
    with open(os.path.join(root_dir, 'ufo/train.json'), 'r') as f:
        t = json.load(f)

    # shutil.copy(os.path.join(root_dir, 'ufo/train.json'), os.path.join(dst_path, 'ufo/train.json'))

    # img_list = os.listdir(os.path.join(root_dir, 'images'))

    img_list = t['images'].keys()
    for i, img_name in enumerate(tqdm(img_list)):
        img_path = os.path.join(root_dir, 'images', img_name)

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        ratio = long_side / max(h, w)
        new_size = int(w * ratio), int(h * ratio)
        img_resize = cv2.resize(img, new_size)
        cv2.imwrite(os.path.join(dst_path, 'images', img_name), img_resize)

        anno = t['images'][img_name]['words']
        for j, v in enumerate(anno.values()):
            pts = v['points']
            pts_new = np.array(pts) * ratio
            v['points'] = pts_new.tolist()

    with open(os.path.join(dst_path, 'ufo/train.json'), 'w') as f:
        json.dump(t, f)