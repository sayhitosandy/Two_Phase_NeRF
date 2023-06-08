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
from dataloader import *

def pretext(nerf, device, conf):
    lr = 5e-3
    optimizer = optim.Adam(nerf.F_c.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Set up train data
    num_of_files, images, poses, init_ds, init_o = set_up_train_data(device, conf)
    img_shp = images.shape[0]

    pretext_losses = []
    num_iters = conf["num_iters_first"]
    # display_every = conf["img_output_every"]
    nerf.F_c.train()  # Set to train mode

    # Pretext Training
    for i in tqdm(range(num_iters)):
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
            torch.save(nerf, f'./{conf["model"]}/pretext_model.pt')
            np.savez(f'./{conf["model"]}/pretext_loss.npz', pretext_loss=np.array(pretext_losses))
        torch.cuda.empty_cache()

    torch.save(nerf, f'./{conf["model"]}/pretext_model.pt')
    np.savez(f'./{conf["model"]}/pretext_loss.npz',pretext_loss=np.array(pretext_losses))
    return nerf

def downstream(nerf, device, conf):
    lr = 5e-3
    optimizer = optim.Adam(nerf.F_c.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Set up test data
    test_folder = conf["test_dir"]
    test_files = os.listdir(test_folder)
    test_image_set, test_pose_set, test_init_ds_set, test_init_o_set = load_data(
        device, os.path.join(test_folder, test_files[0]))
    test_ds, test_os, test_img, train_idxs = set_up_test_data(
        test_image_set, device, test_pose_set, test_init_ds_set, test_init_o_set, conf)
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
            plt.savefig(f'{conf["model"]}/results/Iteration{i}')
            nerf.F_c.train()

        np.savez(f'{conf["model"]}/downstream_metrics.npz',
            epochs=np.array(num_iters),
            train_loss=np.array(losses),
            val_loss=np.array(val_losses),
            psnr=np.array(psnrs))

    print("Done!")

