import os

import torch
from torch.utils.data import Dataset as dataset
import SimpleITK as sitk
import random


class Dataset(dataset):
    def __init__(self, mr_dir, seg_dir, crop, cropsize):
        """

        :param mr_dir: mr数据的地址
        :param seg_dir: 金标准的地址
        """
        self.crop = crop
        self.size = cropsize

        self.mr_list = os.listdir(mr_dir)
        self.seg_list = list(map(lambda x: x.replace('breast', 'label'), self.mr_list))

        self.mr_list = list(map(lambda x: os.path.join(mr_dir, x), self.mr_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):
        """

        :return: torch.Size([B, 1, 48, 256, 256])
        """

        mr_path = self.mr_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        mr = sitk.ReadImage(mr_path)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        mr_array = sitk.GetArrayFromImage(mr)
        seg_array = sitk.GetArrayFromImage(seg)

        if self.crop is True:
            start_slice = random.randint(0, mr_array.shape[0] - self.size)
            end_slice = start_slice + self.size - 1

            mr_array = mr_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]

        # 处理完毕，将array转换为tensor
        mr_array = torch.FloatTensor(mr_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return mr_array, seg_array

    def __len__(self):
        return len(self.mr_list)

# if crop is False:
#     mr_dir = "/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/train_F/MR"
#     seg_dir = "/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/train_F/GT_one"
# else:
#     mr_dir = "/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test/MR"
#     seg_dir = "/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test/GT"
#
# breastData = Dataset(mr_dir, seg_dir)


# # 测试代码
# from torch.utils.data import DataLoader
# train_ds=Dataset("I:/OneDrive - bit.edu.cn/WenZhou/data_20201120_modify/normal_data/test_train/img",
#                  "I:/OneDrive - bit.edu.cn/WenZhou/data_20201120_modify/normal_data/test_train/label",
#                  False,48)
# print(train_ds)
# train_dl = DataLoader(train_ds, 1, True, num_workers=2)
# for index, (mr, seg) in enumerate(train_dl):
#     print(index, mr.size(), seg.size())
#     print('----------------')