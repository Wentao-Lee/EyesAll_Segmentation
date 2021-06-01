import os
from time import time
import random
import numpy as np
import train_config


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import VNet
import Dice

from dataloaders.Eyesall_Dataset_center import Dataset
import SimpleITK as sitk

import visdom

args = train_config.get_args()

torch.autograd.set_detect_anomaly(True)


# 建立保存路径
if not os.path.exists(args["module_save_dir"]):
    os.makedirs(args["module_save_dir"])

# 显卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
# torch.backends.cudnn.benchmark=True

# 设置visdom
vis = visdom.Visdom(env=args["vis"], port=8888)

# 固定种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    # 设置随机数种子
    setup_seed(0)
    # net = DialResUNet(True).cuda()

    # # 定义网络
    net = VNet.VNet(
        n_channels=args["model"]["n_channels"],
        n_classes=args["model"]["n_classes"],
        normalization=args["model"]["normalization"]
    ).cuda()

    # net = VNet(n_channels=1, n_classes=1, normalization='instancenorm', has_dropout=False)
    # net = net.cuda()
    print("net total parameters hhhh:", sum(param.numel() for param in net.parameters()))
    net.train()

    # 定义优化器
    if args["optim"] == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args["lr"])
    elif args["optim"] == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=args["lr"], momentum=0.99, weight_decay=5e-4)

    # 定义数据集

    train_ds = Dataset(
        args["train_dataset"]["train_mr"],
        args["train_dataset"]["train_label"],
        args["train_dataset"]["crop"],
        args["train_dataset"]["cropsize"],
    )

    trainloader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True)

    # 定义损失函数
    # loss_fpce = loss.CrossEntropy()
    #
    # loss_bpce = loss.negCrossEntropy()
    #
    # loss_size = loss.NaivePenalty()
    #
    # loss_ce = nn.BCELoss()
    # # loss
    # loss_dice = loss.DiceLoss()
    # loss_wdice = loss.WDiceLoss()
    #
    # loss_hinge = loss.SpatialEmbLoss()
    #
    # loss_distance = loss.distance_loss()
    #
    # loss_mse = nn.MSELoss()
    #
    # loss_ncut = loss.NCutLoss3D()

    # 学习率衰减
    if args["lr_decay"] == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    elif args["lr_decay"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 开始训练
    start = time()
    t = args["loss_opts"]["t"]

    for epoch in range(args["n_epochs"]):


        mean_loss = []

        for step, data in enumerate(trainloader):

            volume_batch, label_batch = data
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = net(volume_batch)

            # if args["loss_opts"]["loss"] == "fpce+size":
            #     fpceloss = loss_fpce(outputs, label_batch)
            #     sizeloss = args["loss_opts"]["w_size"] * loss_size(outputs, label_batch, d, t)
            #     loss = fpceloss + sizeloss
            #
            # elif args["loss_opts"]["loss"] == "ce":
            #     log_p = (outputs + 1e-10).log()
            #     mask = label_batch.unsqueeze(0).type(torch.float32)
            #
            #     loss1 = (
            #         -0.99 * torch.einsum("bcwhl,bcwhl->", mask, log_p)
            #         - 0.01 * torch.einsum("bcwhl,bcwhl->", (1 - mask), (1 - outputs + 1e-10).log())
            #     ) / (256 * 256 * 128)
            #     # loss=-((mask*log_p+(1-mask)*(1-outputs + 1e-10).log())).sum()/mask.sum()
            #
            #     loss2 = loss_dice(outputs, label_batch)
            #     # outputs=outputs.squeeze(1)
            #     # loss=loss_ce(outputs, label_batch)
            #     loss = loss1 + loss2
            #     # loss=loss_hinge(outputs, label_batch)
            #     # print(loss)
            #

            # print(outputs.shape, label_batch.shape)


            loss_CrossEntropyLoss = nn.CrossEntropyLoss()
            loss_fn = loss_CrossEntropyLoss(outputs, label_batch.long())
            loss = loss_fn
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if step % 10 is 0:
                print(
                    "epoch:{}, step:{}, loss:{:.3f},learning_rate:{:.5f}, time:{:.3f} min".format(
                        epoch,
                        step,
                        loss.item(),
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        (time() - start) / 60,
                    )
                )
                # if args["loss_opts"]["loss"] == "fpce+size":
                #     print(
                #         "epoch:{}, step:{}, loss:{:.3f},loss_fpce:{:.3f},loss_size:{:.3f},learning_rate:{:.5f}, time:{:.3f} min".format(
                #             epoch,
                #             step,
                #             loss.item(),
                #             fpceloss.item(),
                #             sizeloss.item(),
                #             optimizer.state_dict()["param_groups"][0]["lr"],
                #             (time() - start) / 60,
                #         )
                #     )
                #
                # elif args["loss_opts"]["loss"] == "ce":
                #     print(
                #         "epoch:{}, step:{}, loss:{:.3f},learning_rate:{:.5f}, time:{:.3f} min".format(
                #             epoch,
                #             step,
                #             loss.item(),
                #             optimizer.state_dict()["param_groups"][0]["lr"],
                #             (time() - start) / 60,
                #         )
                #     )

        # if args["loss_opts"]["loss"] == "fpce+size":
        #     vis.line(
        #         X=torch.FloatTensor([epoch]),
        #         Y=torch.FloatTensor([fpceloss]),
        #         win="fpce loss",
        #         update="append" if epoch > 1 else None,
        #         opts={"title": "fpce loss"},
        #     )
        #     vis.line(
        #         X=torch.FloatTensor([epoch]),
        #         Y=torch.FloatTensor([sizeloss]),
        #         win="size loss",
        #         update="append" if epoch > 1 else None,
        #         opts={"title": "size loss"},
        #     )

        vis.line(
            X=torch.FloatTensor([epoch]),
            Y=torch.FloatTensor([loss]),
            win="loss",
            update="append" if epoch > 1 else None,
            opts={"title": "train loss"},
        )
        # vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([loss1]), win='train',
        #          update='append' if epoch > 1 else None,opts={'title': 'dice loss'})
        # vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([loss2]), win='train',
        #          update='append' if epoch > 1 else None,opts={'title': 'ce loss'})

        t = t * 1.1

        mean_loss = sum(mean_loss) / len(mean_loss)

        if epoch % 10 is 0:
            # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
            save_mode_path = os.path.join(
                args["module_save_dir"], "net_{}-{:.3f}-{:.3f}.pth".format(epoch, loss.item(), mean_loss)
            )
            torch.save(net.state_dict(), save_mode_path)

        ##测试
        dice_eyeball_list = []
        dice_muscle_list = []
        dice_nerve_list = []
        for file_index, file in enumerate(os.listdir(args["val_dataset"]["val_mr"])):

            volume = sitk.ReadImage(os.path.join(args["val_dataset"]["val_mr"], file))
            volume_array = sitk.GetArrayFromImage(volume)

            with torch.no_grad():
                volume_tensor = torch.FloatTensor(volume_array).cuda()
                volume_tensor = volume_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

                # if args["loss_opts"]["loss"] == "fpce+size+mse":
                #     outputs, outputs_dm = net(volume_tensor)
                #
                # elif args["loss_opts"]["loss"] == "fpce+size+hinge":
                #     outputs, offset = net(volume_tensor)
                # else:
                #     outputs = net(volume_tensor)
                # # outputs = outputs[1,...]

                outputs = net(volume_tensor)

            pred_seg = outputs
            # print(pred_seg.shape)
            pred_seg = outputs.squeeze().cpu().detach().numpy()
            pred_seg = (pred_seg).astype(np.uint8)
            # print(pred_seg.shape)

            seg = sitk.ReadImage(
                os.path.join(args["val_dataset"]["val_gt"], file), sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            # seg_array=ndimage.zoom(seg_array,(2,2,2),order=0)
            # dice = (2 * pred_seg * seg_array).sum() / (pred_seg.sum() + seg_array.sum())
            dice_eyeball = Dice.calculate_dice(pred_seg, seg_array, target_value=1)
            dice_muscle = Dice.calculate_dice(pred_seg, seg_array, target_value=2)
            dice_nerve = Dice.calculate_dice(pred_seg, seg_array, target_value=3)
            print("file: {}, dice_eyeball: {:.3f}".format(file, dice_eyeball))
            print("file: {}, dice_muscle: {:.3f}".format(file, dice_muscle))
            print("file: {}, dice_nerve: {:.3f}".format(file, dice_nerve))
            dice_eyeball_list.append(dice_eyeball)
            dice_muscle_list.append(dice_muscle)
            dice_nerve_list.append(dice_nerve)

        mean_dice_eyeball = sum(dice_eyeball_list) / len(dice_eyeball_list)
        mean_dice_muscle = sum(dice_muscle_list) / len(dice_muscle_list)
        mean_dice_nerve = sum(dice_nerve_list) / len(dice_nerve_list)

        print("mean dice eyeball: {}".format(sum(dice_eyeball_list) / len(dice_eyeball_list)))
        print("mean dice muscle: {}".format(sum(dice_muscle_list) / len(dice_muscle_list)))
        print("mean dice nerve: {}".format(sum(dice_nerve_list) / len(dice_nerve_list)))

        vis.line(
            X=torch.FloatTensor([epoch]),
            Y=torch.FloatTensor([mean_dice_eyeball]),
            win="test",
            name='loss_eyeball',
            update="append" if epoch > 1 else None,
            opts={"title": "test eyeball dice"},
        )

        vis.line(
            X=torch.FloatTensor([epoch]),
            Y=torch.FloatTensor([mean_dice_muscle]),
            win="test",
            name='loss_muscle',
            update="append" if epoch > 1 else None,
            opts={"title": "test muscle dice"},
        )

        vis.line(
            X=torch.FloatTensor([epoch]),
            Y=torch.FloatTensor([mean_dice_nerve]),
            win="test",
            name='loss_nerve',
            update="append" if epoch > 1 else None,
            opts={"title": "test nerve dice"},
        )
