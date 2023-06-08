# !/usr/bin/env python
# coding: utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(device, data_f):
    """
    Load data from npz file
    :param device: Device CPU/GPU
    :param data_f: Data file (.npz)
    :return: images, poses, init_ds, init_o
    """
    data = np.load(data_f)

    images = data["images"] / 255
    img_size = images.shape[1]
    xs = (torch.arange(img_size) - (img_size / 2 - 0.5)).float()
    ys = (torch.arange(img_size) - (img_size / 2 - 0.5)).float()
    (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
    focal = float(data["focal"])
    pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
    camera_coords = pixel_coords / focal
    init_ds = camera_coords.to(device)
    init_o = torch.Tensor(np.array([0, 0, 2.25])).to(device)

    return images, data["poses"], init_ds, init_o


def set_up_train_data(device, conf):
    train_folder = conf["train_dir"]
    train_files = os.listdir(train_folder)
    num_of_files = len(train_files)

    images, poses, init_ds, init_o = None, None, None, None
    for i in range(num_of_files):
        image_set, pose_set, init_ds_set, init_o_set = load_data(device, os.path.join(train_folder, train_files[i]))
        init_ds_set = init_ds_set.unsqueeze(0)
        init_o_set = init_o_set.unsqueeze(0)
        if i == 0:
            images = torch.tensor(image_set)
            poses = torch.tensor(pose_set)
            init_ds = torch.tensor(init_ds_set)
            init_o = torch.tensor(init_o_set)
        else:
            images = torch.cat((images, torch.tensor(image_set)), dim=0)
            poses = torch.cat((poses, torch.tensor(pose_set)), dim=0)
            init_ds = torch.cat((init_ds, torch.tensor(init_ds_set)), dim=0)
            init_o = torch.cat((init_o, torch.tensor(init_o_set)), dim=0)
    return num_of_files, images, poses, init_ds, init_o


def set_up_test_data(images, device, poses, init_ds, init_o, conf, test_idx=150):
    """
    Set up test data
    :param images: Images
    :param device: Device CPU/GPU
    :param poses: Poses
    :param init_ds: Ray directions
    :param init_o: Ray origins
    :param test_idx: Test file index for visualization
    :return: test_ds, test_os, test_img, train_idxs
    """
    plt.imshow(images[test_idx])
    plt.savefig(f'{conf["model"]}/results/test_img.png')

    test_img = torch.Tensor(images[test_idx]).to(device)
    test_R = torch.Tensor(poses[test_idx, :3, :3]).to(device)
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)

    train_idxs = np.arange(len(images)) != test_idx

    return test_ds, test_os, test_img, train_idxs
