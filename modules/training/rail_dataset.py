"""
Rail dataset: loads images from a folder where each image's filename is its timestamp.
Randomly samples pairs per __getitem__, resizes to training_res, converts to grayscale.
"""
import os
import random
import cv2
import torch
from torch.utils.data import Dataset
_IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
class RailDataset(Dataset):
    """
    Dataset that reads all images from *root_dir*, sorts by filename (timestamp),
    and yields random pairs resized to *training_res* (W, H) as grayscale tensors.
    Each sample is a dict with keys:
        image0 : Tensor [1, 1, H, W]   (grayscale, float32, [0,1])
        image1 : Tensor [1, 1, H, W]
    """
    def __init__(self, root_dir, training_res=(800, 608), length=1000):
        """
        Parameters
        ----------
        root_dir : str
            Folder containing images whose filenames are timestamps.
        training_res : tuple (W, H)
            Output resolution to resize images to.
        length : int
            Virtual epoch length (number of pairs per epoch).
        """
        super().__init__()
        self.root_dir = root_dir
        self.training_res = training_res  # (W, H)
        self.length = length
        # Collect and sort image paths
        all_files = sorted(os.listdir(root_dir))
        self.image_paths = [
            os.path.join(root_dir, f) for f in all_files
            if os.path.splitext(f)[1].lower() in _IMG_EXTENSIONS
        ]
        if len(self.image_paths) < 2:
            raise RuntimeError(
                f"[RailDataset] Need at least 2 images in {root_dir}, found {len(self.image_paths)}."
            )
        print(f"[RailDataset] Found {len(self.image_paths)} images in {root_dir}")
    def __len__(self):
        return self.length
    def _load_image(self, path):
        """Read, resize to training_res, convert to grayscale uint8 tensor [1,1,H,W]."""
        w_new, h_new = self.training_res
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"[RailDataset] Could not read image: {path}")
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Ensure im_gray is [H,W] and not empty
        if im_gray is None or im_gray.size == 0:
            raise RuntimeError(f"[RailDataset] Grayscale conversion failed for image: {path}")
        im_gray = cv2.resize(im_gray, (w_new, h_new))
        if im_gray.ndim != 2:
            raise RuntimeError(f"[RailDataset] Grayscale image is not 2D for image: {path}")
        t = torch.from_numpy(im_gray).unsqueeze(0).unsqueeze(0)  # [1,1,H,W], uint8
        return t
    def __getitem__(self, idx):
        # Sample two distinct random images
        i0, i1 = random.sample(range(len(self.image_paths)), 2)
        im0 = self._load_image(self.image_paths[i0])
        im1 = self._load_image(self.image_paths[i1])
        return {"image0": im0, "image1": im1}
