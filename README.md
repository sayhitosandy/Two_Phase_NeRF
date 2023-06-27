# A Two-Phase Training Approach To Boost NeRF Reconstruction Speed
#### Spring 2023, CSE 252D: Advanced Computer Vision Project, UC San Diego

## Installation
Please create a virtual environment:-
```commandline
python3 -m venv two_nerf
source ./two_nerf/bin/activate
```

Install the libraries:
```commandline
python3 -m pip install -r ./requirements.txt
```

## Data
You can find the `.npz` files in `dataset` folder. Each `.npz` file contains 800 images of an object taken from the ShapeNet
dataset. There are different types of objects, such as caps, tables, cars, etc. You can copy the 
required `.npz` file to train folder using the following:
```commandline
cp ./dataset/cars.npz ./data/train/
```
This is only an example. Feel free to use one or more `.npz` files to train the Two-Phase NeRF. 

## Configuration Settings
We can use the `conf.yaml` file to set up all required training and testing parameters. Following describes the configuration parameter.

1. `model` - Determines the type of NeRF we want to train. `tiny_nerf` indicates original tiny nerf module and `two_phase_nerf` indicates the nerf module we introduced.

2. `train_dir` - Path to the directory which holds `.npz` files used for training. It can contain one or multiple files.

3. `test_dir` - Path to the directory which holds `.npz` file used for testing. It can contain only one file. 

4. `test_only` - When set to `True`, we load the saved weights for pretext model and perform only downstream task.

5. `num_iters_first` - Sets the number of run iterations for pretext/ first training in two phase module.

6. `num_iters_second` - Sets the number of run iterations for object specific training in two phase module (second phase). It also indicates the iterations for tiny_nerf.

7. `img_output_every` - Indicates the frequency at which test view validation occurs and stores output and PSNR plots in `model/results` folder.

## Configuration Files
The following configuration files have been set up for training:

1. `conf_baseline.yaml` - Trains the baseline tiny_nerf.
2. `conf.yaml` - Trains a two_phase_nerf with a single category pretext and downstream training.
3. `conf_test.yaml` - Performs downstream training on two_phase_nerf with single category pretext
4. `conf_multi_cat.yaml` - Trains a two_phase_nerf with a multi category pretext and downstream training.

## Train and Test
Please create the `data/train`, `data/test` folders and choose the `conf.yaml` file with desired parameters.

To run the code:
```commandline
python3 ./main.py conf.yaml
```

## Results
Results can be viewed in `{model}/results`. The model stores, image output and PSNR plots at `img_output_every` intervals.

## Additional Utilities
We have added a few utility functions in `utils/`. The files are as follows:

1. `generate_npz_from_shapenet_data` - Generates `.npz` files from ShapeNet dataset.
2. `visualize_shapenet_images` - Generates images from `.npz` files for visualization.

## Contributors
1. Krish Rewanth Sevuga Perumal
2. Manas Sharma
3. Ritika Kishore Kumar
4. Sanidhya Singal

### [Report](https://github.com/sayhitosandy/Two_Phase_NeRF/blob/master/NeRF_Project_Report.pdf)
### [Project Video](https://drive.google.com/file/d/1EhKgIa5kKkrdCRV-tMHIqjBYW0Bi4rND/view?usp=sharing)
