#!/usr/bin/env python
# coding: utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm import tqdm

from nerf import VeryTinyNeRF


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


def set_up_train_data(device):
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


def set_up_test_data(images, device, poses, init_ds, init_o, test_idx=150):
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
    plt.savefig(f'{conf["output_images"]}/test_img.png')

    test_img = torch.Tensor(images[test_idx]).to(device)
    test_R = torch.Tensor(poses[test_idx, :3, :3]).to(device)
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)

    train_idxs = np.arange(len(images)) != test_idx

    return test_ds, test_os, test_img, train_idxs


def run():
    # Set seeds
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load NeRF
    nerf = VeryTinyNeRF(device)

    # Set up nerf
    lr = 5e-3
    optimizer = optim.Adam(nerf.F_c.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if not conf['test_only']:
        # Set up train data
        num_of_files, images, poses, init_ds, init_o = set_up_train_data(device)
        img_shp = images.shape[0]

        pretext_losses = []
        num_iters = conf["num_iters_first"]
        # display_every = conf["img_output_every"]
        nerf.F_c.train()  # Set to train mode

        # Pretext Training
        for _ in tqdm(range(num_iters)):
            idx = np.random.randint(img_shp)
            ds_o_idx = int(idx / (img_shp / num_of_files))
            target_img = images[idx].to(device).float()
            target_pose = poses[idx].to(device)
            target_init_ds = init_ds[ds_o_idx].to(device)
            target_init_o = init_o[ds_o_idx].to(device)
            R = target_pose[:3, :3].float().to(device)
            ds = torch.einsum("ij,hwj->hwi", R, target_init_ds)
            o = (R @ target_init_o).expand(ds.shape)

            C_rs_c = nerf(ds, o)
            loss = criterion(C_rs_c, target_img)
            pretext_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if((i+1)%1000==0):
                store_model_path = conf["save_pretext_model"]
                torch.save(C_rs_c, '{}pretext{}.pt'.format(store_model_path, i))
            torch.cuda.empty_cache()

        store_model_path = conf["load_pretext_model"]
        torch.save(C_rs_c, '{}\pretext_model.pt'.format(store_model_path))

    elif conf['test_only']:
        if conf['load_weights']:
            load_mode_path = conf["load_pretext_model"] + '\pretext_model.pt'
            nerf = torch.load(load_mode_path)
            nerf.F_c.eval()

        # Set up test data
        test_folder = conf["test_dir"]
        test_files = os.listdir(test_folder)
        test_image_set, test_pose_set, test_init_ds_set, test_init_o_set = load_data(
            device, os.path.join(test_folder, test_files[0]))
        test_ds, test_os, test_img, train_idxs = set_up_test_data(
            test_image_set, device, test_pose_set, test_init_ds_set, test_init_o_set)
        images = torch.tensor(test_image_set[train_idxs])
        poses = torch.tensor(test_pose_set[train_idxs])

        # Downstream Training
        psnrs = []
        iternums = []
        losses = []
        val_losses = []
        num_iters = conf["num_iters_second"]
        display_every = conf["img_output_every"]
        nerf.F_c.train()
        # image training - 1
        for i in tqdm(range(num_iters)):
            target_img_idx = np.random.randint(images.shape[0])
            target_pose = poses[target_img_idx].to(device)
            R = target_pose[:3, :3].float().to(device)

            ds = torch.einsum("ij,hwj->hwi", R, test_init_ds_set)
            o = (R @ test_init_o_set).expand(ds.shape)

            C_rs_c = nerf(ds, o)
            loss = criterion(C_rs_c, images[target_img_idx].float().to(device))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            # Visualize the test image with PSNR plot
            if i % display_every == 0:
                nerf.F_c.eval()
                with torch.no_grad():
                    C_rs_c = nerf(test_ds, test_os)

                loss = criterion(C_rs_c, test_img)
                val_losses.append(loss.item())
                psnr = -10.0 * torch.log10(loss)

                psnrs.append(psnr.item())
                iternums.append(i)

                plt.figure(figsize=(10, 4))
                plt.subplot(121)
                plt.imshow(C_rs_c.detach().cpu().numpy())
                plt.title(f"Iteration {i}")
                plt.subplot(122)
                plt.plot(iternums, psnrs)
                plt.title("PSNR")
                plt.savefig(f'{conf["output_images"]}/Iteration{i}')
                nerf.F_c.train()

        np.savez(f'{conf["output_images"]}/tiny_nerf_test_run.npz',
                epochs=np.array(num_iters),
                pretext_loss=np.array(pretext_losses),
                train_loss=np.array(losses),
                val_loss=np.array(val_losses),
                psnr=np.array(psnrs))

        print("Done!")


    # Read configuration
    with open("./conf.yaml", "r") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

    # Train and predict NeRF
    run()
