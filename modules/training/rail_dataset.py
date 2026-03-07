"""
Rail dataset: loads images from a folder where each image's filename is its timestamp.
Randomly samples pairs per __getitem__, converts to grayscale at native resolution
(padded to a multiple of 32 for XFeat compatibility).
"""
import os
import random
import cv2
import torch
from torch.utils.data import Dataset
_IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
def _pad_to_multiple(img, multiple=32):
    """Pad a 2-D array (H, W) on the right/bottom so both dims are divisible by *multiple*."""
    import numpy as np
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
class RailDataset(Dataset):
    """
    Dataset that reads all images from *root_dir*, sorts by filename (timestamp),
    and yields random pairs at **native resolution** as grayscale tensors.
    Images are padded (right/bottom) to the nearest multiple of 32 so that
    XFeat's stride-32 backbone works without cropping artefacts.
    Each sample is a dict with keys:
        image0 : Tensor [1, 1, H, W]   (grayscale, uint8)
        image1 : Tensor [1, 1, H, W]
    """
    def __init__(self, root_dir, length=1000):
        """
        Parameters
        ----------
        root_dir : str
            Folder containing images whose filenames are timestamps.
        length : int
            Virtual epoch length (number of pairs per epoch).
        """
        super().__init__()
        self.root_dir = root_dir
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
        """Read at native resolution, convert to grayscale, pad to mult of 32, return uint8 tensor [1,1,H,W]."""
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"[RailDataset] Could not read image: {path}")
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if im_gray is None or im_gray.size == 0:
            raise RuntimeError(f"[RailDataset] Grayscale conversion failed for image: {path}")
        if im_gray.ndim != 2:
            raise RuntimeError(f"[RailDataset] Grayscale image is not 2D for image: {path}")
        im_gray = _pad_to_multiple(im_gray, 32)
        t = torch.from_numpy(im_gray).unsqueeze(0).unsqueeze(0)  # [1,1,H,W], uint8
        return t
    def __getitem__(self, idx):
        # Sample two distinct random images
        i0, i1 = random.sample(range(len(self.image_paths)), 2)
        im0 = self._load_image(self.image_paths[i0])
        im1 = self._load_image(self.image_paths[i1])
        return {"image0": im0, "image1": im1}
