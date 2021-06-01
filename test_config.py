import os 
import copy


args=dict(
    gpu='7',

    batch_size=1,

    model_dir='I:/OneDrive - bit.edu.cn/WenZhou/module/net_2200-0.004-0.004.pth',

    evaluate=False,

    model={
        'network': 'VNet',
        'n_channels': 1,
        'n_classes': 4,
        'normalization': 'instancenorm',
        # 'dropout': False,
        'pretrain': False,
        # 'multi': False,

    },

    dataset={


        # 'test_mr': 'I:/OneDrive - bit.edu.cn/WenZhou/data_20201120_modify/normal_data/test_normal_crop_xyz/img',
        'test_mr': 'I:/OneDrive - bit.edu.cn/WenZhou/data_20201120_modify/normal_data/test_normal_crop_xyz/img',
        #'/mnt2/mxq/breast/sion_japan/data/OneTumor/train/MR',
        #'/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test/MR'
        #'/mnt2/mxq/breast/sion_japan/data/256*192*96/test/MR'

        'test_label': 'I:/OneDrive - bit.edu.cn/WenZhou/data_20201120_modify/normal_data/test_normal_crop_xyz/label',
        #'/mnt2/mxq/breast/sion_japan/data/256*192*96/test/GT'

        'test_save': 'I:/OneDrive - bit.edu.cn/WenZhou/module/predict',
        #'/mnt2/mxq/breast/sion_japan/data/OneTumor/train/PRED',
        #'/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test/PRED'
        #'/mnt2/mxq/breast/sion_japan/data/256*192*96/test/PRED'

        # 'cropsize':128,
        #
        # 'test_full_mr': "/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test_full/MR_512",
        # 'test_full_label': "/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test_full/GT_512",
        # 'test_full_save': "/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test_full/PRED",

    },
)

def get_args():
    return copy.deepcopy(args)


