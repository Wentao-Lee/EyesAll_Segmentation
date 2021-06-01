import os
from time import time
import sys

sys.path.append('..')
# import test_config
import test_config

import numpy as np
import SimpleITK as sitk
from skimage import measure
import scipy.ndimage as ndimage
import xlsxwriter as xw
import torch
from networks.network import *

args = test_config.get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

workbook = xw.Workbook(os.path.join(args['dataset']['test_save'], 'result.xlsw'))
worksheet = workbook.add_worksheet('net_10-3.091-2.248.pth')

center_bold = workbook.add_format()
center_bold.set_bold()
center_bold.set_align('center')

worksheet.set_column(0, 2, width=50, cell_format=center_bold)
worksheet.write(0, 0, 'file_name')
worksheet.write(0, 1, 'dice')
worksheet.write(0, 2, 'mean_dice')

max = False

###建立预测存储路径
if not os.path.exists(args['dataset']['test_save']):
    os.makedirs(args['dataset']['test_save'])

# 定义网络并加载参数
net = VNet(n_channels=args['model']['n_channels'],
           n_classes=args['model']['n_classes'],
           normalization=args['model']['normalization'],
           # has_dropout=args['model']['dropout'],
           pretrain=args['model']['pretrain'],
           # multi=args['model']['multi']
           ).cuda()

net.load_state_dict(torch.load(args['model_dir']))
net.eval()
dice_list = []
time_list = []

# 开始正式进行测试
for file_index, file in enumerate(os.listdir(args['dataset']['test_mr'])):

    start_time = time()

    # 读入数据
    volume = sitk.ReadImage(os.path.join(args['dataset']['test_mr'], file))
    volume_array = sitk.GetArrayFromImage(volume)

    with torch.no_grad():
        volume_tensor = torch.FloatTensor(volume_array).cuda()
        volume_tensor = volume_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

        outputs = net(volume_tensor)

        # outputs,outputs_dm = net(volume_tensor)

        pred_seg = outputs.squeeze().cpu().detach().numpy()

    # ###连通域后处理
    # if max is True:
    #     pred_seg = measure.label((pred_seg>=1), 4)

    #     areas = [r.area for r in measure.regionprops(pred_seg)]
    #     areas.sort()
    #     print(areas)
    #     c=measure.regionprops(pred_seg)
    #     print(c[0].centroid)
    #     for region in measure.regionprops(pred_seg):
    #         if region.area < 100:   ###连通域选择
    #             for coordinates in region.coords:
    #                 pred_seg[coordinates[0], coordinates[1],coordinates[2]] = 0
    #     pred_seg = pred_seg > 0

    #####最大连通域
    # props = measure.regionprops(pred_seg)
    # max_area = 0
    # max_index = 0
    # for index, prop in enumerate(props, start=1):
    #     if prop.area > max_area:
    #         max_area = prop.area
    #         max_index = index
    #
    # pred_seg[pred_seg != max_index] = 0
    # pred_seg[pred_seg == max_index] = 1

    pred_seg = (pred_seg).astype(np.uint8)  # 1
    # print(pred_seg.shape)

    if args['evaluate'] == True:
        ##读入金标准，计算Dice
        seg = sitk.ReadImage(os.path.join(args['dataset']['test_label'], file.replace('breast', 'label')),
                             sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        dice = (2 * pred_seg * seg_array).sum() / (pred_seg.sum() + seg_array.sum())
        worksheet.write(file_index + 1, 0, file)
        worksheet.write(file_index + 1, 1, dice)
        print('file: {}, dice: {:.3f}'.format(file, dice))
        dice_list.append(dice)

    # 将预测的结果保存为nii数据
    # pred_seg=ndimage.zoom(pred_seg,(0.5,0.5,0.5),order=0)
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(volume.GetDirection())
    pred_seg.SetOrigin(volume.GetOrigin())
    pred_seg.SetSpacing(volume.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(args['dataset']['test_save'], file.replace('breast', 'pred')))
    del pred_seg

    casetime = time() - start_time
    time_list.append(casetime)

    print('file: {} use {:.3f} s'.format(file, casetime))
    print('-----------------------')

# 输出整个测试集平均处理时间
print('time per case: {}'.format(sum(time_list) / len(time_list)))
print('mean dice: {}'.format(sum(dice_list) / len(dice_list)))
worksheet.write(1, 2, sum(dice_list) / len(dice_list))

workbook.close()