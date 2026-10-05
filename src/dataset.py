import os

import numpy as np
import torch
import torchvision.transforms as T
from skimage import io
from torch.utils.data import Dataset


split_imgs = {
    "train": [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26],
    "val": [28, 30, 32, 34, 37],
    "test": [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38],
}


class VaihingenDataset(Dataset):
    def __init__(self, img_folder, GT_folder, split, patch_size=512):
        self.imgs = []
        self.GTs = []

        conversion = T.ToTensor()
        overlap = patch_size // 2

        for img_index in split_imgs[split]:
            print("Working on image " + str(img_index))

            img = (
                io.imread(
                    os.path.join(
                        img_folder,
                        "top_mosaic_09cm_area" + str(img_index) + ".tif",
                    )
                )
                / 255
            )

            GT = io.imread(
                os.path.join(
                    GT_folder,
                    "top_mosaic_09cm_area" + str(img_index) + ".tif",
                )
            )

            for i in np.arange(
                patch_size // 2,
                img.shape[0] - patch_size // 2,
                overlap,
            ):
                for j in np.arange(
                    patch_size // 2,
                    img.shape[1] - patch_size // 2,
                    overlap,
                ):
                    self.imgs.append(
                        conversion(
                            img[
                                i - patch_size // 2 : i + patch_size // 2,
                                j - patch_size // 2 : j + patch_size // 2,
                                :,
                            ]
                        )
                    )

                    self.GTs.append(
                        GT[
                            i - patch_size // 2 : i + patch_size // 2,
                            j - patch_size // 2 : j + patch_size // 2,
                        ]
                    )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].float()
        GT = self.GTs[idx]

        return img, torch.from_numpy(GT)