import os 
import SimpleITK as sitk
import numpy as np
import nibabel as nib

def dcm2nii(dcms_path, nii_path):
	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
	# 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z

    print(image_array.min(), image_array.max())
	
    # 3.将array转为img，并保存为.nii.gz
    # image_array = np.flip(image_array, axis = 1)
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)

def read_data(path):
    img=nib.load(path)
    img_array = img.get_fdata()#channel last,存放图像数据的矩阵 
    affine_array = img.affine.copy()#get the affine array, 定义了图像数据在参考空间的位置
    img_head = img.header.copy(); #get image metadat, 图像的一些属性信息，采集设备名称，体素的大小，扫描层数
    #获取其他一些信息的方法
    img.shape # 获得维数信息
    img.get_data_dtype() # 获得数据类型
    img_head.get_data_dtype() #获得头信息的数据类型
    img_head.get_data_shape()# 获得维数信息
    img_head.get_zooms() #获得体素大小

    return img_array,affine_array,img_head


if __name__ == '__main__':

    old_img_path = "G:/YJX_Data/Baseline_ATD_data/img"
    img_path = "G:/YJX_Data/Abodominal_Case/img"
    old_mask_path = "G:/YJX_Data/Abodominal_Case/mask"

    data_path = "G:/YJX_Data/RSNA_ATD_data/train_images/"  # dicom序列文件所在路径
    save_path = "G:/YJX_Data/Baseline_ATD_data/img_v2"  # 所需.nii.gz文件保存路径
    msksave_path = "G:/YJX_Data/Baseline_ATD_data/mask_v2"
    filelist = []

    for oldfilename in os.listdir(old_img_path):   # 已有的image
        filelist.append(oldfilename)
    
    count = 0
    for filename in os.listdir(old_mask_path):

        patientid = filename.split('.')[0].split('_')[0]
        seriesid = filename.split('.')[0].split('_')[1]
        print(patientid, seriesid)

        if filename not in filelist:

            dcms_path = os.path.join(data_path, patientid, seriesid)
            nii_path = os.path.join(save_path, filename)
            mask_path = os.path.join(old_mask_path, filename)
            msknii_path = os.path.join(msksave_path, filename)

            print(dcms_path, nii_path)
            dcm2nii(dcms_path, nii_path)

            _,affine_array,img_head = read_data(nii_path)
            msk = nib.load(mask_path)
            msk_array = msk.get_fdata().astype(np.uint8)#channel last,存放图像数据的矩阵 
            msk_array = np.flip(msk_array, axis = 1)

            new_nii = nib.Nifti1Image(msk_array,affine_array,img_head)
            nib.save(new_nii,msknii_path)
        
        count = count + 1


