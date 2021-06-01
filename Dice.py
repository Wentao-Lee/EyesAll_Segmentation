import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib


def calculate_dice(pre, label, target_value):
    if target_value == 1:
        pre[pre == 2] = 0
        pre[pre == 3] = 0
        label[label == 2] = 0
        label[label == 3] = 0

    if target_value == 2:
        pre[pre == 1] = 0
        pre[pre == 2] = 1
        pre[pre == 3] = 0
        label[label == 1] = 0
        label[label == 2] = 1
        label[label == 3] = 0

    if target_value == 3:
        pre[pre == 1] = 0
        pre[pre == 2] = 0
        pre[pre == 3] = 1
        label[label == 1] = 0
        label[label == 2] = 0
        label[label == 3] = 1

    intersection = np.logical_and(pre, label)
    intersection = np.sum(intersection)

    union = np.sum(label) + np.sum(pre)

    dice = 2 * intersection / union
    return dice

# data_path = 'I:/OneDrive - bit.edu.cn/WenZhou/data_20201120_modify/normal_data/test'
# # data_img = nib.load(os.path.join(data_path, '001.nii.gz'))
# # data_array = data_img.get_fdata()
# # data_array[data_array == 2] = 0
# # data_array[data_array == 3] = 0
# # nib.save(nib.Nifti1Image(data_array.astype('uint8'), None, data_img.header), os.path.join(data_path, '001new.nii.gz'))
# data = sitk.ReadImage(os.path.join(data_path, '001.nii.gz'))
# data_array = sitk.GetArrayFromImage(data)
# data_array[data_array == 2] = 2
# data_array[data_array == 3] = 0
# data_img = sitk.GetImageFromArray(data_array)
# sitk.WriteImage(data_img, os.path.join(data_path, '001newsitk.nii.gz'))