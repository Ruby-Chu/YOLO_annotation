import os
import pandas as pd
import numpy as np
from PIL import Image
import shutil


def load_bbx(bbx_path):
    annotations = {}
    with open(bbx_path, mode='r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            file_name = lines[i].strip()
            i += 1
            num_boxes = int(lines[i].strip())
            i += 1
            boxes = []
            for _ in range(num_boxes):
                box_info = lines[i].strip().split()
                box = {
                    'x': int(box_info[0]),
                    'y': int(box_info[1]),
                    'w': int(box_info[2]),
                    'h': int(box_info[3]),
                }
                boxes.append(box)
                i += 1
            if num_boxes == 0:
                i += 1
            if num_boxes > 0:
                annotations[file_name] = boxes

    return annotations


def annotation_to_df(annotation, img_shape):
    cs, xs, ys, ws, hs = [], [], [], [], []

    h, w, _ = img_shape
    for box in annotation:
        cs.append(0)
        xs.append((box["x"] + box["w"] / 2.0) / w)
        ys.append((box["y"] + box["h"] / 2.0) / h)
        ws.append(box["w"] / w)
        hs.append(box["h"] / h)

    return pd.DataFrame({0: cs, 1: xs, 2: ys, 3: ws, 4: hs})


def add_dataset(keys, annotations, img_folder, root, split):
    if not os.path.exists(os.path.join(root, "images", split)):
        os.makedirs(os.path.join(root, "images", split))
    if not os.path.exists(os.path.join(root, "labels", split)):
        os.makedirs(os.path.join(root, "labels", split))

    for i, key in enumerate(keys):
        img_path = os.path.join(img_folder, key)
        copy_img_path = os.path.join(root, "images", split, "im{}.jpg".format(i))
        shutil.copyfile(img_path, copy_img_path)
        img = np.array(Image.open(img_path))
        df = annotation_to_df(annotations[key], img.shape)
        df.to_csv(os.path.join(root, "labels", split, "im{}.txt".format(i)), header=False, index=False, sep=' ')


if __name__ == "__main__":
    train_img_folder = "WIDER_train/images/"
    val_img_folder = "WIDER_val/images/"
    test_img_folder = "WIDER_test/images/"
    ann_folder = "wider_face_split"
    ann_train_file_name = "wider_face_train_bbx_gt.txt"
    ann_val_file_name = "wider_face_val_bbx_gt.txt"
    train_annotations = load_bbx(os.path.join(ann_folder, ann_train_file_name))
    val_annotations = load_bbx(os.path.join(ann_folder, ann_val_file_name))
    test_list_file = os.path.join(ann_folder, "wider_face_test_filelist.txt")

    train_keys = []
    val_keys = []
    for key in train_annotations.keys():
        train_keys.append(key)
    for key in val_annotations.keys():
        val_keys.append(key)

    add_dataset(train_keys, train_annotations, train_img_folder, "WIDER_dataset/", "train")
    add_dataset(val_keys, val_annotations, val_img_folder, "WIDER_dataset/", "val")

    # move test dataset
    if not os.path.exists(os.path.join("WIDER_dataset", "test")):
        os.makedirs(os.path.join("WIDER_dataset", "test"))

    with open(test_list_file, mode='r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            img_path = os.path.join(test_img_folder, line.strip())
            copy_path = os.path.join("WIDER_dataset", "test", "im{}.jpg".format(i))
            shutil.copyfile(img_path, copy_path)
