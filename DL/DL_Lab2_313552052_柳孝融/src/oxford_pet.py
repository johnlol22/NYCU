import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from scipy.ndimage import rotate
from torchvision import transforms

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}
        np.random.seed(220)
        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        # self defined preprocess
        image, mask = self.augment_sample(image, mask)

        
        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)
    
    def adjust_brightness_contrast(self, img, brightness_factor=0.2, contrast_factor=0.2):
        brightness = 1.0 + np.random.uniform(-brightness_factor, brightness_factor)
        contrast = 1.0 + np.random.uniform(-contrast_factor, contrast_factor)
        img = img * brightness
        img = (img - 0.5) * contrast + 0.5
        return np.clip(img, 0, 1)

    def augment_sample(self, image, mask):
        # Store original dtype to convert back at the end
        original_dtype = image.dtype
    
        # Convert to float for processing
        if original_dtype != np.float32 and original_dtype != np.float64:
            image = image.astype(np.float32) / 255.0
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # Random rotation (small angles)
        angle = np.random.uniform(-15, 15)
        # For scipy.ndimage.rotate
        image = rotate(image, angle, reshape=False, mode='nearest')
        mask = rotate(mask, angle, reshape=False, mode='nearest', order=0)

        # Random brightness/contrast adjustment (image only)
        image = self.adjust_brightness_contrast(image)

        # Convert back to original dtype if needed
        if original_dtype == np.uint8:
            image = (image * 255).astype(np.uint8)

        return image, mask
    def calculate_dataset_stats(self, dataset_path):
        all_image_paths = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_image_paths.append(os.path.join(root, file))

        means = []
        stds = []
        print("Calculating dataset statistics...")
        for image_path in tqdm(all_image_paths):
            try:
                img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
                means.append(np.mean(img, axis=(0, 1)))
                stds.append(np.std(img, axis=(0, 1)))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        dataset_mean = np.mean(means, axis=0)
        dataset_std = np.mean(stds, axis=0)
        print(f"Dataset mean: {dataset_mean}")
        print(f"Dataset std: {dataset_std}")
        return dataset_mean, dataset_std




class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode):
    # implement the load dataset function here
    """
    Load the Oxford Pet dataset for the specified mode
    
    Args:
        data_path (str): Root directory where the dataset is located or will be downloaded to
        mode (str): One of 'train', 'valid', or 'test'
        
    Returns:
        SimpleOxfordPetDataset: The loaded dataset ready for use with PyTorch DataLoader
    """
    # Check if dataset exists, if not, download it
    if not os.path.exists(os.path.join(data_path, "images")) or \
       not os.path.exists(os.path.join(data_path, "annotations")):
        print(f"Dataset not found in {data_path}. Downloading...")
        OxfordPetDataset.download(data_path)
        print("Download completed.")
    
    # Create and return the dataset object
    return SimpleOxfordPetDataset(root=data_path, mode=mode)
    