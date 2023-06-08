#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import yaml

from nerf import VeryTinyNeRF
from traning import *

def main(conf):
    # Set seeds
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load NeRF
    nerf = VeryTinyNeRF(device)

    if not os.path.exists(f'{conf["model"]}'):
        os.makedirs(f'{conf["model"]}')
    if not os.path.exists(f'{conf["model"]}/results'):
        os.makedirs(f'{conf["model"]}/results')

    if conf['model'] == 'tiny_nerf':
        nerf = downstream(nerf, device, conf)
    elif conf['model'] == 'two_phase_nerf' and conf['test_only']:
        nerf = torch.load(f'{conf["model"]}/pretext_model.pt')
        nerf.F_c.eval()
        nerf = downstream(nerf, device, conf)
    else:
        nerf = pretext(nerf, device, conf)
        nerf = downstream(nerf, device, conf)


if __name__ == '__main__':    
    # Read configuration
    with open("./conf.yaml", "r") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

    # Train and predict NeRF
    main(conf)
