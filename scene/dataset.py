from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            item = self.dataset[index]

            # Colmap/scene_reader entries already carry correct filename metadata.
            if hasattr(item, "R") and hasattr(item, "image"):
                caminfo = item
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask
                image_name = getattr(caminfo, "image_name", f"{index}")
                image_path = getattr(caminfo, "image_path", None)
            else:
                image, w2c, time = item
                R, T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask = None
                image_name = f"{index}"
                image_path = None

            camera = Camera(
                colmap_id=index,
                R=R,
                T=T,
                FoVx=FovX,
                FoVy=FovY,
                image=image,
                gt_alpha_mask=None,
                image_name=image_name,
                uid=index,
                data_device=torch.device("cuda"),
                time=time,
                mask=mask,
            )
            if image_path is not None:
                camera.image_path = image_path
            return camera
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
