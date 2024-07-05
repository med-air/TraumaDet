from torch.utils.data import Dataset
import numpy as np 
from monai.transforms import Randomizable, apply_transform
from monai.utils import MAX_SEED
import torch

class FusionDataset(Dataset, Randomizable):
    def __init__(self, local_npz_files, global_npz_files, labels=None, local_img_transforms=None, local_seg_transforms=None, global_img_transforms=None, global_seg_transforms=None, isTraining=False) -> None:
        super().__init__()
        self.local_npz_files = local_npz_files
        self.labels    = labels 
        self.local_img_transforms = local_img_transforms
        self.local_seg_transforms = local_seg_transforms

        self.global_npz_files = global_npz_files
        self.global_img_transforms = global_img_transforms
        self.global_seg_transforms = global_seg_transforms

        self.isTraining = isTraining
    
    def __len__(self) -> int:
        return len(self.local_npz_files)

    def randomize(self) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()

        batch_data = {}
        # load npz_file
        local_data = np.load(self.local_npz_files[index])
        liver      = local_data['liver']
        spleen     = local_data['spleen']
        left_kidney = local_data['left_kidney']
        right_kidney = local_data['right_kidney']

        global_data = np.load(self.global_npz_files[index])
        abdominal = global_data['image']

        if self.local_img_transforms is not None:
            if isinstance(self.local_img_transforms, Randomizable):
                self.local_img_transforms.set_random_state(seed=self._seed)
            
            liver       = apply_transform(self.local_img_transforms, liver, map_items=False)
            spleen       = apply_transform(self.local_img_transforms, spleen, map_items=False)
            left_kidney = apply_transform(self.local_img_transforms, left_kidney, map_items=False)
            right_kidney = apply_transform(self.local_img_transforms, right_kidney, map_items=False)
        
        if self.global_img_transforms is not None:
            if isinstance(self.global_img_transforms, Randomizable):
                self.global_img_transforms.set_random_state(seed=self._seed)
            
            abdominal = apply_transform(self.global_img_transforms, abdominal, map_items=False)

            
        if self.labels is not None:
            label = self.labels[index]
            batch_data["label"] = label


        batch_data['name']  = self.local_npz_files[index]
        batch_data['liver'] = liver
        batch_data['spleen'] = spleen
        batch_data['left_kidney'] = left_kidney
        batch_data['right_kidney'] = right_kidney

        batch_data['abdominal'] = abdominal


        return batch_data