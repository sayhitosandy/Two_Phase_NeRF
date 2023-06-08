def create_images(data_folder):
    import os

    import numpy as np

    from PIL import Image
    from tqdm import tqdm

    images_folder = "../data/images"
    files = os.listdir(data_folder)

    for file in files:
        data = np.load(f"{folder}/{file}")
        filename = file.split(".")[0]

        if not os.path.exists(f"{images_folder}/{filename}"):  # Create folder if it does not exist
            os.makedirs(f"{images_folder}/{filename}")

        for i in tqdm(range(len(data["images"]))):
            img = Image.fromarray(data["images"][i], "RGB")
            img.save(f"{images_folder}/{filename}/image_{i}.png")


if __name__ == "__main__":
    folder = "../data/train/"  # data folder containing .npz file(s)
    create_images(data_folder=folder)
