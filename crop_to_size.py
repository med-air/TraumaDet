import numpy as np 
import os 
from skimage.measure import label 
from scipy.ndimage import zoom
import nibabel as nib

def get_crop_range(seg):
    # find the seg bbox
    x_start, x_end = np.where(np.any(seg, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(seg, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(seg, axis=(0, 1)))[0][[0, -1]]

    x_start, x_end = max(0, x_start), min(seg.shape[0], x_end)
    y_start, y_end = max(0, y_start), min(seg.shape[1], y_end)
    z_start, z_end = max(0, z_start), min(seg.shape[2], z_end)

    # find the center point
    x_mid = (x_start + x_end) // 2
    y_mid = (y_start + y_end) // 2
    z_mid = (z_start + z_end) // 2
    
    # expand to 96 * 96 * 96
    x_start, x_end = max(0, x_mid - 48), min(seg.shape[0], x_mid + 48)
    y_start, y_end = max(0, y_mid - 48), min(seg.shape[1], y_mid + 48)
    z_start, z_end = max(0, z_mid - 48), min(seg.shape[2], z_mid + 48)

    return (x_start, x_end, y_start, y_end, z_start, z_end)

def crop_liver(liver_seg):
    x_start, x_end = np.where(np.any(liver_seg, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(liver_seg, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(liver_seg, axis=(0, 1)))[0][[0, -1]]


    x_start, x_end = max(0, x_start - 5), min(liver_seg.shape[0], x_end + 5)
    y_start, y_end = max(0, y_start - 5), min(liver_seg.shape[1], y_end + 5)
    z_start, z_end = max(0, z_start - 5), min(liver_seg.shape[2], z_end + 5)

    return (x_start, x_end, y_start, y_end, z_start, z_end)


def make_sure_size(img):
    x_size, y_size, z_size = img.shape
    pad_img = np.pad(img, ((0, 96-x_size), (0, 96-y_size), (0, 96-z_size)), mode="constant", constant_values=0)

    return pad_img

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 )
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def process_single_npz(npz_file: np.ndarray):
    img = npz_file['image'] # 128 * 156 * 130
    seg = npz_file['seg']

    liver_seg  = (seg == 1)
    spleen_seg = (seg == 2)
    lkidney_seg = (seg == 3)
    rkidney_seg = (seg == 4)

    liver_seg = getLargestCC(liver_seg)
    spleen_seg = getLargestCC(spleen_seg)
    lkidney_seg = getLargestCC(lkidney_seg)
    rkidney_seg = getLargestCC(rkidney_seg)

    # crop the liver part
    x_start, x_end, y_start, y_end, z_start, z_end = crop_liver(liver_seg)
    liver_img = img[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_shape = liver_img.shape
    liver_img = zoom(liver_img, (96/liver_shape[0], 96/liver_shape[1], 96/liver_shape[2]))
    

    # crop the spleen part
    x_start, x_end, y_start, y_end, z_start, z_end = get_crop_range(spleen_seg)
    spleen_img = img[x_start:x_end, y_start:y_end, z_start:z_end]    
    spleen_img = make_sure_size(spleen_img)


    # crop the left kidney part
    x_start, x_end, y_start, y_end, z_start, z_end = get_crop_range(lkidney_seg)
    lkidney_img = img[x_start:x_end, y_start:y_end, z_start:z_end]
    lkidney_img = make_sure_size(lkidney_img)

    # crop the right kidney part
    x_start, x_end, y_start, y_end, z_start, z_end = get_crop_range(rkidney_seg)
    rkidney_img = img[x_start:x_end, y_start:y_end, z_start:z_end]
    rkidney_img = make_sure_size(rkidney_img)

    return (liver_img, spleen_img, lkidney_img, rkidney_img)


def main():

    flag = False
    save_root = "G:/YJX_Data/Baseline_Competation_method/all_part_train"
    file_root = "G:/YJX_Data/Baseline_Competation_method/Train_Data_npz"

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    filelist  = os.listdir(file_root)

    for name in filelist:
        print(name)
        # if (name == "64194_25349.npz"):
        #     flag = True

        # if flag:

        single_npz_file = np.load(os.path.join(file_root, name))
        print(name, "is processing")
        liver_img, spleen_img, lkidney_img, rkidney_img = process_single_npz(single_npz_file)

        print(name, liver_img.shape, spleen_img.shape, lkidney_img.shape, rkidney_img.shape)
        # save npz file
        np.savez(os.path.join(save_root, name), liver=liver_img, spleen=spleen_img, left_kidney=lkidney_img, right_kidney=rkidney_img, seg=single_npz_file['seg'])
        
        # else:
        #     continue



if __name__ == "__main__":
    main()