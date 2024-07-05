from monai import transforms, data
import numpy as np 
import os 

def _get_transform():
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            # transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
            transforms.ToTensord(keys=["image", "label"]),
        ]
        )
    
    return train_transform


def find_smallest_box(seg):
    x_start, x_end = np.where(np.any(seg, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(seg, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(seg, axis=(0, 1)))[0][[0, -1]]

    x_start, x_end = max(0, x_start-5), min(seg.shape[0], x_end+5)
    y_start, y_end = max(0, y_start-5), min(seg.shape[1], y_end+5)
    z_start, z_end = max(0, z_start-5), min(seg.shape[2], z_end+5)

    return (x_start, x_end, y_start, y_end, z_start, z_end)

def process(save_root, loader):
    for batch_data in loader:
        inputs, segs = batch_data['image'], batch_data['label']
        inputs = inputs[0,0].numpy()
        segs   = segs[0,0].numpy()  

        name = batch_data["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]
        name = name.split('.')[0].split('\\')[-1]
        x_start, x_end, y_start, y_end, z_start, z_end = find_smallest_box(segs)
        crop_input = inputs[x_start:x_end, y_start:y_end, z_start:z_end]
        crop_seg = segs[x_start:x_end, y_start:y_end, z_start:z_end]

        crop_shape = (x_end-x_start, y_end-y_start, z_end-z_start) 
        print(name, inputs.shape, np.max(inputs), np.min(inputs), crop_shape)

        # save to .npz file
        np.savez(save_root + name, image=crop_input, seg=crop_seg)

def process_train():
    save_root = "G:/YJX_Data/Baseline_Competation_method/Train_Data_npz/"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    datalist_json='training_samples.json'
    datalist = data.load_decathlon_datalist(datalist_json, True, "training")

    transform = _get_transform()
    ds = data.Dataset(data=datalist, transform=transform)
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    process(save_root, loader)


def process_test():
    save_root = "model/tmp_data/monai_preprocess/test"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    datalist_json='my_code/test_samples.json'
    datalist = data.load_decathlon_datalist(datalist_json, True, "test", base_dir='')
    transform = _get_transform()
    ds = data.Dataset(data=datalist, transform=transform)
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    process(save_root, loader)

if __name__ == "__main__":
    process_train()
    # process_test()