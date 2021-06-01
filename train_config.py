import copy
import os


args=dict(
    gpu='6',
    vis='eyesall_20210526_test',

    lr=0.0001,
    lr_decay='step',
    optim='Adam',
    n_epochs=5000,
    batch_size=2,

    # module_save_dir='/mnt/Disk1/liwentao/Vnet/Module/Eyesall_20210530',
    module_save_dir='/mnt2/homes/wentaol/module/Wenzhou/eyesall/20210530_test',

    model={
        'network': 'VNet',
        'n_channels': 1,
        'n_classes': 4,
        'normalization': 'instancenorm',
        'dropout': False

    },

    loss_opts={
        'loss':'fpce+size+distance',
            #'fpce+size+distance',
            #'fpce+size+mse',
            #'fpce+size'
            #'fpce+bpce+size'
            #'TwoStage'
            #'fpce+size'
            #'ce',
        'bbox':False ,
        'w_bpce': 0.0000001,

        'w_ncut': 1,

        'axis':False,

        'dm':False,
        'w_mse':1,

        'hinge':False,
        'w_hinge':1,

        'distance':True,
        'w_distance':0.01,

        'log_barrier': False,
        'w_size': 0.01,
        
        'w_wdice': 0.00001,
        't': 5,

    },

    train_dataset={
        'train_mr': "/mnt2/homes/wentaol/data/Wenzhou/normal_data/train_xyz_data/img",
        # 'train_mr': "/mnt/Disk1/liwentao/data/Wenzhou/normal_data/train_xyz_data/img",


        # 'train_label': "/mnt/Disk1/liwentao/data/Wenzhou/normal_data/train_xyz_data/label",
        'train_label': "/mnt2/homes/wentaol/data/Wenzhou/normal_data/train_xyz_data/label",

  
        # 'train_box': "/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/train_F/GT_ellipsoid",
        # 'train_axis':"/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/train_F/axis.xlsx",
        # 'train_dm':"/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/train_F/GT_DM",

        'crop': False,
        'cropsize': 48,

    },

    val_dataset={
        # 'val_mr': '/mnt/Disk1/liwentao/data/Wenzhou/normal_data/val_xyz_data/img',
        'val_mr': '/mnt2/homes/wentaol/data/Wenzhou/normal_data/val_xyz_data/img',
        #"/mnt2/mxq/breast/sion_japan/data/256*192*96/test/MR"
        #'/mnt2/mxq/promise12/test/MR-norm'
        #'/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test/MR'


        # 'val_gt': '/mnt/Disk1/liwentao/data/Wenzhou/normal_data/val_xyz_data/label',
        'val_gt': '/mnt2/homes/wentaol/data/Wenzhou/normal_data/val_xyz_data/label',
        #"/mnt2/mxq/breast/sion_japan/data/256*192*96/test/GT"
        #'/mnt2/mxq/promise12/test/GT'
        #'/mnt2/mxq/breast/sion_japan/data/256x256x48/z-score/test/GT'
    },

)

def get_args():
    return copy.deepcopy(args)