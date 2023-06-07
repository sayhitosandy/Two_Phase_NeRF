# A Two-Phase Training Approach To Boost NeRF Reconstruction Speed
#### Spring 2023, CSE 252D: Advanced Computer Vision Project, UC San Diego

## Installation
Please create a virtual environment:
```commandline
python3 -m venv two_nerf
source ./two_nerf/bin/activate
```

Install the libraries:
```commandline
python3 -m pip install -r ./requirements.txt
```

## Train and Test
Please create the `data/train`, `data/test`, and `data/output` folders, and update their paths in `config.yaml`.
The train and test folders contain the `.npz` files for training and testing respectively. The output folders stores 
the predicted images after `img_output_every` interval. 

To run the code:
```commandline
python3 ./main.py
```

## Data
You can find the `.npz` files in `dataset` folder. Each `.npz` file contains 800 images of an object taken from the ShapeNet
dataset. There are different types of objects, such as caps, tables, cars, etc. You can copy the 
required `.npz` file to train folder using the following:
```commandline
cp ./dataset/cars.npz ./data/train/
```
This is only an example. Feel free to use one or more `.npz` files to train the Two-Phase NeRF. 

## Contributors
1. Krish Rewanth Sevuga Perumal
2. Manas Sharma
3. Ritika Kishore Kumar
4. Sanidhya Singal
