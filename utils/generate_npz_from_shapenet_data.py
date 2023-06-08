#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from pyrr import Matrix44
from tqdm import tqdm

from renderer import gen_rotation_matrix_from_azim_elev_in_plane, Renderer
from renderer_settings import *


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        # print(f"Directory '{directory_path}' created successfully!")


def main():
    # Set up the renderer.
    renderer = Renderer(
        camera_distance=CAMERA_DISTANCE,
        angle_of_view=ANGLE_OF_VIEW,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    img_size = 100
    # Calculate focal length in pixel units. This is just geometry. See:
    # https://en.wikipedia.org/wiki/Angle_of_view#Derivation_of_the_angle-of-view_formula.
    focal = (img_size / 2) / np.tan(np.radians(ANGLE_OF_VIEW) / 2)

    # Load the ShapeNet car object.
    for i in tqdm(range(len(folder_names))):
        # obj = "a3d8771740fd7630afd6b353b2d4958f"
        for j in range(len(objs[i])):
            obj = objs[i][j]
            cat = folder_names[i]
            obj_mtl_path = f"{folder_path}/{cat}/{obj}/models/model_normalized"
            renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")

            # Generate car renders using random camera locations.
            init_cam_pos = np.array([0, 0, CAMERA_DISTANCE])
            target = np.zeros(3)
            up = np.array([0.0, 1.0, 0.0])
            samps = 800
            imgs = []
            poses = []
            for idx in range(samps):
                angles = {
                    "azimuth": np.random.uniform(-np.pi, np.pi),
                    "elevation": np.random.uniform(-np.pi, np.pi),
                }
                R = gen_rotation_matrix_from_azim_elev_in_plane(**angles)
                eye = tuple((R @ init_cam_pos).flatten())
                look_at = Matrix44.look_at(eye, target, up)
                renderer.prog["VP"].write(
                    (look_at @ renderer.perspective).astype("f4").tobytes()
                )
                renderer.prog["cam_pos"].value = eye

                image = renderer.render(0.5, 0.5, 0.5).resize((img_size, img_size))
                imgs.append(np.array(image))

                pose = np.eye(4)
                pose[:3, :3] = np.array(look_at[:3, :3])
                pose[:3, 3] = -look_at[:3, :3] @ look_at[3, :3]
                poses.append(pose)

            imgs = np.stack(imgs)
            poses = np.stack(poses)
            create_directory(f"Gen\{cat}")
            np.savez(
                f"Gen\{cat}\{obj}.npz",
                images=imgs,
                poses=poses,
                focal=focal,
                camera_distance=CAMERA_DISTANCE,
            )


if __name__ == "__main__":
    # Example usage
    # directory_path = "path/to/directory"
    # create_directory(directory_path=directory_path)

    # Replace with the actual path to the 'image' folder
    folder_path = "../dataset/ShapeNetCore.v2"

    # Get a list of all items (files and folders) within the 'image' folder
    items = os.listdir(folder_path)
    # Filter out the folders from the list
    folder_names = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    objs = []
    for folder in folder_names:
        it = os.listdir(os.path.join(folder_path, folder))
        it = it[:6]
        objs.append(it)
    print(objs)
    print(folder_names)
    main()
