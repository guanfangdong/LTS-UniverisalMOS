import sys
from requests import patch
import torch
import argparse
import os
import random
from skimage.filters import threshold_otsu

import imageio
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import time

from common_py.dataIO import loadFiles_plus
from common_py.dataIO import loadImgs_pytorch
from common_py.dataIO import checkCreateDir

from common_py.evaluationBS import evaluation_numpy_entry_torch
from common_py.evaluationBS import evaluation_numpy_entry
from common_py.evaluationBS import evaluation_numpy
from common_py.evaluationBS import evaluation_BS

# from common_py.utils import setupSeed

from common_py.improcess import img2pos
# from model_freeze import freeze_by_names, unfreeze_by_names
from onet import ONet_TINY, U_Net, ONet_TINY_UPPER, ONet_BASE

from torch import optim


def setupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx):

    fs_im, fullfs_im = loadFiles_plus(pa_im, ft_im)
    fs_fg, fullfs_fg = loadFiles_plus(pa_fg, ft_fg)
    fs_gt, fullfs_gt = loadFiles_plus(pa_gt, ft_gt)

    im = torch.tensor(imageio.imread(fullfs_im[idx])).unsqueeze(0)
    fg = torch.tensor(imageio.imread(fullfs_fg[idx])).unsqueeze(0)
    gt = torch.tensor(imageio.imread(fullfs_gt[idx])).unsqueeze(0)

    imXfg = torch.cat((im, fg.unsqueeze(-1)), dim=-1)

    return imXfg, gt


def getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx_list):

    imXfg, gt = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx_list[0])

    num_list = len(idx_list)

    for i in range(1, num_list):
        idx = idx_list[i]

        imXfg_tmp, gt_tmp = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt,
                                     idx)

        imXfg = torch.cat((imXfg, imXfg_tmp), dim=0)
        gt = torch.cat((gt, gt_tmp), dim=0)


#        print("idx = ", idx)

#    imXfg = imXfg.permute(0, 3, 1, 2)
#    gt = gt.unsqueeze(-1).permute(0, 3, 1, 2)

#     idx = gt < 250
#     gt[idx] = 0

    return imXfg, gt


def test_unet(imgs, device, network, len_batch):

    NUM_DATA = imgs.shape[0]

    NUM_BATCH = round(NUM_DATA / len_batch - 0.5)

    left = 0
    i = 0

    imgs_batch = imgs[left:left + len_batch]
    imgs_batch = imgs_batch.to(device, dtype=torch.float32)

    re_mask = network(imgs_batch).detach().cpu()
    left += len_batch

    for i in range(1, NUM_BATCH):

        imgs_batch = imgs[left:left + len_batch]
        imgs_batch = imgs_batch.to(device, dtype=torch.float32)

        mask = network(imgs_batch).detach().cpu()

        re_mask = torch.cat((re_mask, mask), dim=0)

        left += len_batch

    if left < NUM_DATA:
        imgs_batch = imgs[left:NUM_DATA]
        imgs_batch = imgs_batch.to(device, dtype=torch.float32)

        #         print("left = ", left)
        #         print("NUM_DATA = ", NUM_DATA)
        mask = network(imgs_batch).detach().cpu()

        re_mask = torch.cat((re_mask, mask), dim=0)

    return re_mask


def test_unet_gpu(imgs, device, network, len_batch):

    NUM_DATA = imgs.shape[0]

    NUM_BATCH = round(NUM_DATA / len_batch - 0.5)

    imgs = imgs.to(device, dtype=torch.float32)

    left = 0
    i = 0

    imgs_batch = imgs[left:left + len_batch]
    #    imgs_batch = imgs_batch.to(device, dtype=torch.float32)
    with torch.no_grad():
        re_mask = network(imgs_batch)
    left += len_batch

    for i in range(1, NUM_BATCH):

        imgs_batch = imgs[left:left + len_batch]
        #        imgs_batch = imgs_batch.to(device, dtype=torch.float32)
        with torch.no_grad():
            mask = network(imgs_batch)
        #.detach().cpu()

        re_mask = torch.cat((re_mask, mask), dim=0)

        left += len_batch

    if left < NUM_DATA:
        imgs_batch = imgs[left:NUM_DATA]
        #        imgs_batch = imgs_batch.to(device, dtype=torch.float32)

        #         print("left = ", left)
        #         print("NUM_DATA = ", NUM_DATA)
        with torch.no_grad():
            mask = network(imgs_batch)
        #.detach().cpu()

        re_mask = torch.cat((re_mask, mask), dim=0)

    return re_mask.detach().cpu()


def train_unet(imgs,
               labs,
               device,
               network,
               optimizer,
               loss_func,
               len_batch,
               num_epoch,
               net_pa,
               train_o_net=True):

    NUM_DATA = imgs[2].shape[0]
    NUM_DATAES = [NUM_DATA * 4 * 4, NUM_DATA * 4, NUM_DATA]

    len_batch_list = [len_batch * 4 * 4, len_batch * 4, len_batch]

    NUM_BATCH = round(NUM_DATA / len_batch - 0.5)
    print(NUM_BATCH)

    # NUM_BATCHES = [NUM_BATCH*4*4, NUM_BATCH*4, NUM_BATCH]

    loss_list = []
    #torch.tensor([])

    # fig = plt.figure(figsize=(4, 4))

    # if train_o_net:
    #     ################# train unet first
    #     # train onet first
    #     freeze_by_names(
    #         network,
    #         ["unet_encode", "down1", "down2", "down3", "up1", "up2", "up3"])
    #     #################END#####################

    for i in range(num_epoch):

        idx_data = [
            torch.randperm(imgs[0].shape[0]),
            torch.randperm(imgs[1].shape[0]),
            torch.randperm(imgs[2].shape[0])
        ]

        left_list = [0, 0, 0]

        total_loss = 0

        for j in range(NUM_BATCH):
            for level in range(3):
                len_batch = len_batch_list[level]
                left = left_list[level]
                idx_batch = idx_data[level][left:left + len_batch]

                imgs_batch = imgs[level][idx_batch]
                imgs_batch = imgs_batch.to(device, dtype=torch.float32)

                labs_batch = labs[level][idx_batch]
                labs_batch = labs_batch.to(device, dtype=torch.long)

                mask_pred = network(imgs_batch)

                loss = loss_func(mask_pred, labs_batch)

                print(f"batch{j}/{NUM_BATCH}/{level}")

                print("epoch:", i, "/", num_epoch, "   loss:",
                      loss.detach().cpu().item())

                total_loss += loss.detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                left_list[level] += len_batch

        for level in range(3):
            if left_list[level] < NUM_DATAES[level]:
                # train rest
                left = left_list[level]
                idx_batch = idx_data[level][left:]

                imgs_batch = imgs[level][idx_batch]
                imgs_batch = imgs_batch.to(device, dtype=torch.float32)

                labs_batch = labs[level][idx_batch]
                labs_batch = labs_batch.to(device, dtype=torch.long)

                mask_pred = network(imgs_batch)

                loss = loss_func(mask_pred, labs_batch)

                print("epoch:", i, "/", num_epoch, "   loss:",
                      loss.detach().cpu().item())

                total_loss += loss.detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # idx_batch = idx_data[left:left + len_batch]

            # imgs_batch = imgs[idx_batch]
            # labs_batch = labs[idx_batch]

            # imgs_batch = imgs_batch.to(device, dtype=torch.float32)
            # labs_batch = labs_batch.to(device, dtype=torch.long)

            # mask_pred = network(imgs_batch)

            # # print(mask_pred.shape)
            # # print(labs_batch.shape)
            # # print(mask_pred)
            # # print(labs_batch)

            # loss = loss_func(mask_pred, labs_batch.squeeze())

            # print("epoch:", i, "/", num_epoch, "   loss:",
            #       loss.detach().cpu().item())

            # total_loss += loss.detach().cpu().item()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # left += len_batch

        # if left != NUM_DATA:
        #     idx_batch = idx_data[left:NUM_DATA]

        #     imgs_batch = imgs[idx_batch]
        #     labs_batch = labs[idx_batch]

        #     imgs_batch = imgs_batch.to(device, dtype=torch.float32)
        #     labs_batch = labs_batch.to(device, dtype=torch.long)

        #     mask_pred = network(imgs_batch)

        #     loss = loss_func(mask_pred, labs_batch.squeeze())

        #     #print("loss:", loss.detach().cpu().item())
        #     print("epoch:", i, "/", num_epoch, "   loss:",
        #           loss.detach().cpu().item())

        #     total_loss += loss.detach().cpu().item()

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # save model
        epoch = i
        if epoch % 20 == 0:
            name_net = net_pa + "network_dis_" + str(epoch).zfill(4) + '.pt'

            checkCreateDir(name_net)
            torch.save(network.state_dict(), name_net)

            print("\n\n save model completed")

        loss_list.append(total_loss)
        # write loss to txt
        name_loss = net_pa + "loss.txt"
        with open(name_loss, "a") as f:
            f.write(str(total_loss) + "\n")

        # x = np.linspace(0, i, num=i + 1)
        # y = loss_list

        # plt.plot(x, y, '.:', color=[0.1, 0.1, 0.1])

        # plt.pause(0.01)
    return network


'''
    if train_o_net:
        # unfreeze unet
        unfreeze_by_names(
            network,
            ["unet_encode", "down1", "down2", "down3", "up1", "up2", "up3"])


        ###########Train U-Net then############
        # freeze_by_names(network, [
        #     "onet_encode", "up_o_1", "up_o_2", "up_o_3", "down_o_1",
        #     "down_o_2", "down_o_3"
        # ])
        #################END#####################
        for i in range(num_epoch, int(num_epoch * 1.5)):

            idx_data = torch.randperm(NUM_DATA)

            left = 0

            total_loss = 0

            for j in range(NUM_BATCH):

                idx_batch = idx_data[left:left + len_batch]

                imgs_batch = imgs[idx_batch]
                labs_batch = labs[idx_batch]

                imgs_batch = imgs_batch.to(device, dtype=torch.float32)
                labs_batch = labs_batch.to(device, dtype=torch.long)

                mask_pred = network(imgs_batch)

                # print(mask_pred.shape)
                # print(labs_batch.shape)
                # print(mask_pred)
                # print(labs_batch)

                loss = loss_func(mask_pred, labs_batch.squeeze())

                print("epoch:", i, "/", num_epoch, "   loss:",
                      loss.detach().cpu().item())

                total_loss += loss.detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                left += len_batch

            if left != NUM_DATA:
                idx_batch = idx_data[left:NUM_DATA]

                imgs_batch = imgs[idx_batch]
                labs_batch = labs[idx_batch]

                imgs_batch = imgs_batch.to(device, dtype=torch.float32)
                labs_batch = labs_batch.to(device, dtype=torch.long)

                mask_pred = network(imgs_batch)

                loss = loss_func(mask_pred, labs_batch.squeeze())

                #print("loss:", loss.detach().cpu().item())
                print("epoch:", i, "/", num_epoch, "   loss:",
                      loss.detach().cpu().item())

                total_loss += loss.detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # save model
            epoch = i
            if epoch % 20 == 0:
                name_net = net_pa + "network_dis_" + str(epoch).zfill(
                    4) + '.pt'

                checkCreateDir(name_net)
                torch.save(network.state_dict(), name_net)

                print("\n\n save model completed")

            loss_list.append(total_loss)

            x = np.linspace(0, i, num=i + 1)
            y = loss_list

            plt.plot(x, y, '.:', color=[0.1, 0.1, 0.1])

            plt.pause(0.01)

    return network
'''


################MARK####################
def getPatches(im, center, radius):

    if len(im.shape) < 3:
        im = im.unsqueeze(-1)

    row_im = im.shape[0]
    column_im = im.shape[1]

    tl = center - radius
    tl[tl < 0] = 0

    rb = tl + 2 * radius

    rb[rb[:, 0] > row_im, 0] = row_im
    rb[rb[:, 1] > column_im, 1] = column_im

    tl = rb - 2 * radius

    tl = tl.long()
    rb = rb.long()
    num_patch = tl.shape[0]

    i = 0

    #     print(tl)
    #     print(tl[i, 0])
    #     print(rb[i, 0])
    #     print(tl[i, 1])
    #     print(rb[i, 1])

    re_patches = im[tl[i, 0]:rb[i, 0], tl[i, 1]:rb[i, 1], :].unsqueeze(0)

    for i in range(1, num_patch):

        patches = im[tl[i, 0]:rb[i, 0], tl[i, 1]:rb[i, 1], :].unsqueeze(0)

        re_patches = torch.cat((re_patches, patches), dim=0)

    return re_patches


################MARK####################
def img2patch_plus(im, mask, num_pos, num_neg, radius):

    pos = img2pos(im)

    idx_pos = mask == 255
    idx_neg = mask == 0

    pos_pos = pos[idx_pos, :]
    pos_neg = pos[idx_neg, :]

    LEN_pos = pos_pos.shape[0]
    LEN_neg = pos_neg.shape[0]

    idx = torch.randperm(LEN_pos)
    pos_pos = pos_pos[idx, :]

    idx = torch.randperm(LEN_neg)
    pos_neg = pos_neg[idx, :]

    flag = 0

    if num_pos <= LEN_pos:
        center_pos = pos_pos[0:num_pos, :]

        patches_pos = getPatches(im, center_pos, radius)
        mask_pos = getPatches(mask, center_pos, radius)

        re_patches = patches_pos
        re_mask = mask_pos

        flag += 1

    if num_neg <= LEN_neg:
        center_neg = pos_neg[0:num_neg, :]

        patches_neg = getPatches(im, center_neg, radius)
        mask_neg = getPatches(mask, center_neg, radius)

        re_patches = patches_neg
        re_mask = mask_neg

        flag += 1

    if flag == 2:
        re_patches = torch.cat((patches_pos, patches_neg), dim=0)
        re_mask = torch.cat((mask_pos, mask_neg), dim=0)

    return re_patches, re_mask


def img2patch(im, mask, num_pos, num_neg):
    pos = img2pos(im)

    idx_pos = mask == 255
    idx_neg = mask == 0

    pos_pos = pos[idx_pos, :]
    pos_neg = pos[idx_neg, :]

    LEN_pos = pos_pos.shape[0]
    LEN_neg = pos_neg.shape[0]

    idx = torch.randperm(LEN_pos)
    pos_pos = pos_pos[idx, :]

    idx = torch.randperm(LEN_neg)
    pos_neg = pos_neg[idx, :]

    center_pos = pos_pos[0:num_pos, :]
    center_neg = pos_neg[0:num_neg, :]

    #     print("im.shape = ", im.shape)
    #     print("center_pos = ", center_pos)
    #     print("pos_pos = ", pos_pos)
    #     print("num test = ", pos_pos.shape[0])

    patches_pos = getPatches(im, center_pos, 60)
    patches_neg = getPatches(im, center_neg, 60)

    mask_pos = getPatches(mask, center_pos, 60)
    mask_neg = getPatches(mask, center_neg, 60)

    re_patches = torch.cat((patches_pos, patches_neg), dim=0)
    re_mask = torch.cat((mask_pos, mask_neg), dim=0)

    return re_patches, re_mask


def imgseq2patches(imXfg, gt, num_pos, num_neg, radius):

    patches, mask = img2patch_plus(imXfg[0], gt[0], num_pos, num_neg, radius)

    frames = imXfg.shape[0]

    for i in range(1, frames):
        patches_temp, mask_temp = img2patch_plus(imXfg[i], gt[i], num_pos,
                                                 num_neg, radius)

        patches = torch.cat((patches, patches_temp), dim=0)
        mask = torch.cat((mask, mask_temp), dim=0)

    return patches, mask


def imgseq2patches_seedfill(list_imXfg, list_gt, radius):

    #    starttime = time.time()
    lt_list_im, patches = img2patches_seedfill(list_imXfg[0], radius)
    #    endtime = time.time()
    #    print("time = ", endtime - starttime)

    #     starttime = time.time()
    #     lt_list_im1, patches1 = img2patches_seedfill_fast(list_imXfg[0], radius)
    #     endtime = time.time()
    #     print("time = ", endtime - starttime)

    #     print("borderline ================================")
    #
    #     print(lt_list_im)
    #     print(lt_list_im1)
    #
    #     print(torch.sum(lt_list_im - lt_list_im1))
    #     print(torch.sum(patches - patches1))
    #
    #     print("borderline -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    #
    #
    #
    #     stopins = input()

    #    print(list_imXfg[0].shape)
    #    print(list_gt[0].shape)

    lt_list_gt, mask = img2patches_seedfill(list_gt[0].unsqueeze(-1), radius)

    frames = list_imXfg.shape[0]

    for i in range(1, frames):

        lt_list_im, patches_temp = img2patches_seedfill(list_imXfg[i], radius)
        lt_list_gt, mask_temp = img2patches_seedfill(list_gt[i].unsqueeze(-1),
                                                     radius)

        # patches_temp, mask_temp = img2patch_plus(imXfg[i], gt[i], num_pos, num_neg, radius)

        patches = torch.cat((patches, patches_temp), dim=0)
        mask = torch.cat((mask, mask_temp), dim=0)

    return patches, mask


def poscheck(r, c, row_im, column_im, radius):

    if r + radius > row_im:
        r = row_im - radius

    if c + radius > column_im:
        c = column_im - radius

    return r, c


def patches2img_seedfill(patches, lt_list, radius):

    row_im = torch.max(lt_list[:, 0]) + radius
    column_im = torch.max(lt_list[:, 1]) + radius

    byte_im = 1
    if len(patches.squeeze().shape) == 4:
        byte_im = patches.shape[3]

    re_im = torch.zeros(row_im, column_im, byte_im).squeeze()

    num = lt_list.shape[0]

    for i in range(num):
        r = lt_list[i, 0]
        c = lt_list[i, 1]

        re_im[r:r + radius, c:c + radius] = patches[i].squeeze()

    return re_im


def img2patches_seedfill_slow(im, radius):

    row_im = im.shape[0]
    column_im = im.shape[1]

    canvas = torch.zeros(row_im, column_im)

    lt_list = torch.tensor([0, 0]).unsqueeze(0)

    canvas[0:radius, 0:radius] = 1
    re_patches = im[0:radius, 0:radius, :].unsqueeze(0)

    for r in range(row_im):
        for c in range(column_im):

            if canvas[r, c] == 0:
                r, c = poscheck(r, c, row_im, column_im, radius)

                canvas[r:r + radius, c:c + radius] = 1
                patches_tmp = im[r:r + radius, c:c + radius, :].unsqueeze(0)

                re_patches = torch.cat((re_patches, patches_tmp), dim=0)

                pos = torch.tensor([r, c])
                lt_list = torch.cat((lt_list, pos.unsqueeze(0)), dim=0)

    return lt_list, re_patches


def img2patches_seedfill(im, radius):

    row_im = im.shape[0]
    column_im = im.shape[1]

    len_r = round(row_im / radius + 0.5)
    len_c = round(column_im / radius + 0.5)

    for i in range(len_r):
        for j in range(len_c):
            if (i + j) == 0:
                r, c = poscheck(i * radius, j * radius, row_im, column_im,
                                radius)

                pos = torch.tensor([r, c]).unsqueeze(0)
                lt_list = pos

                re_patches = im[0:radius, 0:radius, :].unsqueeze(0)

            else:

                r, c = poscheck(i * radius, j * radius, row_im, column_im,
                                radius)

                pos = torch.tensor([r, c])
                lt_list = torch.cat((lt_list, pos.unsqueeze(0)), dim=0)

                patches_tmp = im[r:r + radius, c:c + radius, :].unsqueeze(0)

                re_patches = torch.cat((re_patches, patches_tmp), dim=0)

    return lt_list, re_patches


def test_unet_ImXFg(network, device, pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt,
                    idx, radius, len_batch):

    imXfg, gt = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
    imXfg = imXfg.squeeze()

    lt_list, patches = img2patches_seedfill(imXfg, radius)

    patches = patches.permute(0, 3, 1, 2)

    output = test_unet(patches, device, network, len_batch)

    output = output.argmax(dim=1, keepdim=True).cpu().detach().squeeze()

    fgimg = patches2img_seedfill(output, lt_list, radius)

    return fgimg


def videos2patches(pa_im_list, pa_fg_list, pa_gt_list, idx_list, ft_im, ft_fg,
                   ft_gt, radius, num_pos, num_neg):

    # imXfg_winterDriveway, gt_winterDriveway = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx_list_winterDriveway)

    num = 0

    for pa_im in pa_im_list:
        num = num + 1

    i = 0
    pa_im = pa_im_list[i]
    pa_fg = pa_fg_list[i]
    pa_gt = pa_gt_list[i]

    idx = idx_list[i]

    imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
    re_patches, re_mask = imgseq2patches(imXfg, gt, num_pos, num_neg, radius)

    for i in range(1, num):

        pa_im = pa_im_list[i]
        pa_fg = pa_fg_list[i]
        pa_gt = pa_gt_list[i]

        idx = idx_list[i]

        imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)

        patches, mask = imgseq2patches(imXfg, gt, num_pos, num_neg, radius)

        re_patches = torch.cat((re_patches, patches), dim=0)
        re_mask = torch.cat((re_mask, mask), dim=0)

    return re_patches, re_mask


def videos2patches_seedfill(pa_im_list, pa_fg_list, pa_gt_list, idx_list,
                            ft_im, ft_fg, ft_gt, radius):

    # imXfg_winterDriveway, gt_winterDriveway = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx_list_winterDriveway)

    num = 0

    for pa_im in pa_im_list:
        num = num + 1

    i = 0
    pa_im = pa_im_list[i]
    pa_fg = pa_fg_list[i]
    pa_gt = pa_gt_list[i]

    idx = idx_list[i]

    imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
    re_patches, re_mask = imgseq2patches_seedfill(imXfg, gt, radius)

    for i in range(1, num):

        pa_im = pa_im_list[i]
        pa_fg = pa_fg_list[i]
        pa_gt = pa_gt_list[i]

        idx = idx_list[i]

        imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
        patches, mask = imgseq2patches_seedfill(imXfg, gt, radius)

        re_patches = torch.cat((re_patches, patches), dim=0)
        re_mask = torch.cat((re_mask, mask), dim=0)

    return re_patches, re_mask


# def imgseq2patches_seedfill(list_imXfg, list_gt, radius):


def detectFgImg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, radius, idx, network,
                device, len_batch):

    #     idx_list = [    [1700, 1810, 1832, 1850, 1873, 1881],
    #                     [718, 733, 840, 1125, 1129, 1160],
    #                     [805, 843, 1426, 2764, 2814]]
    #
    #
    #
    #
    #     i = 0
    #     pa_im = pa_im_list[i]
    #     pa_fg = pa_fg_list[i]
    #     pa_gt = pa_gt_list[i]
    #
    #     idx = idx_list[i]

    #     imXfg, gt = getImXFgSeq(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
    #
    #     print(imXfg[0].shape)
    #
    #
    #     idx = 1700

    print("pa_im: ", pa_im, "ft_im: ", ft_im, "pa_fg: ", pa_fg, "ft_fg: ",
          ft_fg, "pa_gt: ", pa_gt, "ft_gt: ", ft_gt, "idx: ", idx)

    imXfg, gt = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)
    #    radius = 128
    #    print(imXfg.shape)

    imXfg = imXfg.squeeze()
    gt = gt.squeeze()

    starttime = time.time()
    lt_list, patches = img2patches_seedfill(imXfg, radius)
    endtime = time.time()
    print("img2patches_seedfill:", endtime - starttime)

    #    print(imXfg.shape)
    patches = patches.permute(0, 3, 1, 2)
    patches = patches.float() / 255.0

    #    len_batch = 5
    starttime = time.time()
    re_mask = test_unet_gpu(patches, device, network, len_batch)
    endtime = time.time()

    print("test_unet_gpu:", endtime - starttime)

    starttime = time.time()
    refinefg = patches2img_seedfill(re_mask.permute(0, 2, 3, 1), lt_list,
                                    radius)
    endtime = time.time()
    print("patches2img_seedfill:", endtime - starttime)

    # refinemask = re_mask.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()
    refinemask = refinefg.argmax(dim=2, keepdim=True).cpu().detach().squeeze()

    return refinemask, gt, imXfg, patches


def detect_current_level(img_patch_bags, gt_patch_bags, original_centers,
                         network, len_batch, current_radius, original_gt):
    '''
    img_patch_bags: 4d tensor, [num_patches, 4, patch_size, patch_size]
    gt_patch_bags: 4d tensor, [num_patches, 1, patch_size, patch_size]
    original_centers: 2d list, save the original center of each patch
    network: the trained network
    len_batch: the batch size
    current_radius: the radius of the current level
    original_gt: the original ground truth 2d tensor
    '''
    # img_patch_bags to float32

    NUM_DATA = img_patch_bags.shape[0]
    NUM_BATCH = round(NUM_DATA / len_batch - 0.5)

    imgs = img_patch_bags.to("cuda", dtype=torch.float32)
    imgs = imgs / 255.0

    left = 0
    i = 0

    imgs_batch = imgs[left:left + len_batch]
    with torch.no_grad():
        re_mask = network(imgs_batch)
    left += len_batch

    for i in range(1, NUM_BATCH):
        imgs_batch = imgs[left:left + len_batch]
        with torch.no_grad():
            mask = network(imgs_batch)

        re_mask = torch.cat((re_mask, mask), dim=0)
        left += len_batch

    if left < NUM_DATA:
        imgs_batch = imgs[left:NUM_DATA]
        with torch.no_grad():
            mask = network(imgs_batch)

        re_mask = torch.cat((re_mask, mask), dim=0)

    re_mask = re_mask.argmax(dim=1, keepdim=True).detach().squeeze()
    # generate a buffer to save the refined mask
    sampled_mask = torch.zeros(original_gt.shape, dtype=torch.int64).cuda()
    refined_mask = torch.zeros(original_gt.shape, dtype=torch.int64).cuda()
    # buffer padding by radius
    pad = transforms.Pad(current_radius, padding_mode='reflect')
    # extend one dimension for the refined mask
    refined_mask = refined_mask.unsqueeze(0)
    refined_mask = pad(refined_mask)
    refined_mask = refined_mask.squeeze(0)

    sampled_mask = sampled_mask.unsqueeze(0)
    sampled_mask = pad(sampled_mask)
    sampled_mask = sampled_mask.squeeze(0)
    
    for idx, pos in enumerate(original_centers):
        x, y = pos
        x += current_radius
        y += current_radius
        refined_mask[x - current_radius:x + current_radius,
                     y - current_radius:y + current_radius] += re_mask[idx]
        sampled_mask[x - current_radius:x + current_radius,
                     y - current_radius:y + current_radius] += 1

    refined_mask = refined_mask[current_radius:-current_radius,
                                current_radius:-current_radius]
    sampled_mask = sampled_mask[current_radius:-current_radius,
                                current_radius:-current_radius]


    return refined_mask, sampled_mask


def detectFgImg_random(pa_im,
                       ft_im,
                       pa_fg,
                       ft_fg,
                       pa_gt,
                       ft_gt,
                       idx,
                       network,
                       device,
                       len_batch,
                       layers,
                       dataset,
                    #    num_pos=[512 * 4 * 4, 512 * 4, 512],
                       radius=[16, 32, 64]):
    '''
    pa_im: path for input data
    ft_im: format for input data
    pa_fg: path for foreground data
    ft_fg: format for foreground data
    pa_gt: path for ground truth data
    ft_gt: format for ground truth data
    radius: radius of the patch, eg: [16, 32, 64]
    idx: index of the frame
    network: network for detection
    device: device for detection
    len_batch: batch size for detection
    '''

    print(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx, sep='\n')


    if (dataset == "CDNet2014"):
        idx = str(idx).zfill(6)
        input_path = "{}in{}.{}".format(pa_im, idx, ft_im)
        fg_path = "{}gt{}.{}".format(pa_fg, idx, ft_fg)
        gt_path = "{}gt{}.{}".format(pa_gt, idx, ft_gt)
    elif (dataset == "LASIESTA"):
        idx = str(idx+1).zfill(6)
        video_name = pa_im.split("/")[-2]
        input_path = "{}{}-{}.{}".format(pa_im, video_name, idx, ft_im)
        fg_path = "{}{}-GT_{}.{}".format(pa_fg, video_name, idx, ft_fg)
        gt_path = "{}{}-GT_{}.{}".format(pa_gt, video_name, idx, ft_gt)

    '''
    imXfg: (row_im, column_im) tensor
    img: (row_im, column_im, channel) tensor
    gt: (row_im, column_im) tensor
    num_pos: number of patches
    radius: radius of patches
    '''
    # read image

    fg = imageio.imread(fg_path)
    img = imageio.imread(input_path)
    gt = imageio.imread(gt_path)

    # convert to tensor
    img = torch.from_numpy(img)
    fg = torch.from_numpy(fg)
    gt = torch.from_numpy(gt)

    # print shape
    print("img.shape: ", img.shape)
    print("fg.shape: ", fg.shape)
    print("gt.shape: ", gt.shape)

    refined_mask = torch.zeros(gt.shape, dtype=torch.int64).cuda()
    sampled_mask = torch.zeros(gt.shape, dtype=torch.int64).cuda()

    refine_mask_different_level = []

    for level in range(len(radius)):
        current_num_pos = int(((img.shape[0] * img.shape[1]) * layers) / ((2 * radius[level]) ** 2))
        current_radius = radius[level]
        img_patch_bags, gt_patch_bags, imXfg_patch_bags, original_centers = cut_patches_randomly_updated_test_use(
            fg, img, gt, current_num_pos, current_radius)

        imXfg_patch_bags = imXfg_patch_bags.unsqueeze(1)
        img_patch_bags = torch.cat((img_patch_bags, imXfg_patch_bags), 1)

        current_refined_mask, current_sampled_mask = detect_current_level(
            img_patch_bags, gt_patch_bags, original_centers, network,
            len_batch, current_radius, gt)



        refined_mask += current_refined_mask
        sampled_mask += current_sampled_mask
        refine_mask_different_level.append(current_refined_mask)

    # print whole array

    hist_mask = refined_mask / sampled_mask
    hist_mask = hist_mask.cpu().numpy()
    # print("hist_mask.shape: ", hist_mask.shape)
    # if nan in hist_mask, replace it with 0
    hist_mask[np.isnan(hist_mask)] = 0

    # thresh = threshold_otsu(hist_mask)
    refinemask = hist_mask > 0.5

    # imXfg, gt = getImXFg(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, idx)

    # imXfg = imXfg.squeeze()
    # gt = gt.squeeze()

    # starttime = time.time()
    # lt_list, patches = img2patches_seedfill(imXfg, radius)
    # endtime = time.time()
    # print("img2patches_seedfill:", endtime - starttime)

    # patches = patches.permute(0, 3, 1, 2)
    # patches = patches.float() / 255.0

    # #    len_batch = 5
    # starttime = time.time()
    # re_mask = test_unet_gpu(patches, device, network, len_batch)
    # endtime = time.time()

    # print("test_unet_gpu:", endtime - starttime)

    # starttime = time.time()
    # refinefg = patches2img_seedfill(re_mask.permute(0, 2, 3, 1), lt_list,
    #                                 radius)
    # endtime = time.time()
    # print("patches2img_seedfill:", endtime - starttime)

    # refinemask = refinefg.argmax(dim=2, keepdim=True).cpu().detach().squeeze()

    return hist_mask, refinemask, fg, gt, refine_mask_different_level


def save_figures(Fm,
                 Fm_bef,
                 fgim,
                 rfim,
                 gtim,
                 i,
                 vid_cat,
                 vid_name,
                 visual_output,
                 type="savetime"):

    # if type == "complex":
    #     plt.figure(figsize=(20, 8))

    #     plt.subplot(1, 3, 1)
    #     plt.imshow(fgim, cmap='gray')
    #     # title keep 4 decimal places
    #     plt.title("Fm_bef: %.4f" % Fm_bef)

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(rfim, cmap='gray')
    #     plt.title("Fm: %.4f" % Fm)

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(gtim, cmap='gray')
    #     plt.title("gt")

    #     # save the plot, file path: results_visual/vid_cat/vid_name/frames.png
    #     # create the path if not exist
    #     pa_visual = "results_visual/" + vid_cat + "/" + vid_name + "/"
    #     if not os.path.exists(pa_visual):
    #         os.makedirs(pa_visual)
    #     plt.savefig(
    #         "results_visual/" + vid_cat + "/" + vid_name + "/" + str(i) +
    #         ".png")
    #     plt.close()
    # elif type == "savetime":
    pa_visual = visual_output + vid_cat + "/" + vid_name + "/"
    if not os.path.exists(pa_visual):
        os.makedirs(pa_visual)
    # save fgim, titled "Fm_bef: %.4f" % Fm_bef
    imageio.imwrite(
        visual_output + vid_cat + "/" + vid_name + "/" +
        "%d_Fm_bef_%.4f.png" % (i, Fm_bef), fgim)
    # save rfim, titled "Fm: %.4f" % Fm
    imageio.imwrite(
        visual_output + vid_cat + "/" + vid_name + "/" +
        "%d_Fm_%.4f.png" % (i, Fm), rfim)
    # save gtim, titled "gt"
    imageio.imwrite(
        visual_output + vid_cat + "/" + vid_name + "/" + "%d_gt.png" % i,
        gtim)


def checkTrainList(idx, list_idx):

    idx_t = torch.tensor(idx)
    list_idx_t = torch.tensor(list_idx)

    list_sub = list_idx_t - idx_t

    judge = torch.sum(list_sub == 0)

    return judge


def evaluateRefinement_random_sample(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt,
                                     radius, network, device, len_batch,
                                     list_train, flag_in, f, vid_cat, vid_name,
                                     save_fig, test_layer, visual_output, dataset):

    sum_TP = 0
    sum_FP = 0
    sum_TN = 0
    sum_FN = 0

    patch_size = [16,32,64]

    sum_TP_rf = 0
    sum_FP_rf = 0
    sum_TN_rf = 0
    sum_FN_rf = 0

    fs, fullfs = loadFiles_plus(pa_gt, ft_gt)
    frames = len(fullfs)

    plt.figure(figsize=(4, 4))
    for i in range(frames):

        starttime = time.time()

        mask_gt = torch.tensor(imageio.imread(fullfs[i]), dtype=torch.float)

        judge_flag = torch.sum(mask_gt == 255) + torch.sum(mask_gt == 0)

        judge_train = checkTrainList(i, list_train)

        if judge_train != 0 and flag_in == 1:
            judge_flag = 0
            print("This frame is included in training set:", i)
            print("list_train:", list_train)

        if judge_flag == 0:
            refinemask = mask_gt
        else:
            #            starttime = time.time()

            print("processing frame: {}/{}".format(i, frames), end='\r')

            hist_mask, refinemask, imXfg, gt, all_refine_mask = detectFgImg_random(
                pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, i, network, device,
                len_batch, test_layer, dataset)
            print(hist_mask)
            refinemask = refinemask * 1
            # hist_mask is np array cast hist_mask to int



            # save hist_mask
            pa_visual =visual_output + vid_cat + "/" + vid_name + "/"
            if not os.path.exists(pa_visual):
                os.makedirs(pa_visual)
            imageio.imwrite(
                visual_output + vid_cat + "/" + vid_name + "/" +
                "%d_hist_mask.png" % i, hist_mask)
            for j, each_mask in enumerate(all_refine_mask):
                each_mask = each_mask.cpu().numpy()
                imageio.imwrite(
                    visual_output + vid_cat + "/" + vid_name + "/" +
                    "%d_hist_mask_patch_%d.png" % (i, patch_size[j]), each_mask)

            #            endtime = time.time()

            #            print("network processing time:", endtime - starttime)

            # srim = imXfg[:, :, 0:3].int()
            fgim = imXfg.int()
            # refinemask to torch
            refinemask = torch.tensor(refinemask, dtype=torch.float)
            rfim = (refinemask * 255).int()
            gtim = gt.int()

            TP, FP, TN, FN = evaluation_numpy_entry_torch(fgim, gtim)

            Re = TP / max((TP + FN), 1)
            Pr = TP / max((TP + FP), 1)

            Fm = (2 * Pr * Re) / max((Pr + Re), 0.0001)

            Re_bef = Re
            Pr_bef = Pr
            Fm_bef = Fm

            print("\n")
            print(
                "=========================================================================================="
            )
            print("pa_im:", pa_im)
            print(fullfs[i])
            print("frames:", i)
            print("current original:")
            print("Re:", Re)
            print("Pr:", Pr)
            print("Fm:", Fm)

            sum_TP += TP
            sum_FP += FP
            sum_TN += TN
            sum_FN += FN

            TP, FP, TN, FN = evaluation_numpy_entry_torch(rfim, gtim)

            #TP, FP, TN, FN = evaluation_numpy_entry(rfim.numpy(), gtim.numpy())

            Re = TP / max((TP + FN), 1)
            Pr = TP / max((TP + FP), 1)

            Fm = (2 * Pr * Re) / max((Pr + Re), 0.0001)

            print("\n current refinefg:")
            print("Re:", Re)
            print("Pr:", Pr)
            print("Fm:", Fm)

            sum_TP_rf += TP
            sum_FP_rf += FP
            sum_TN_rf += TN
            sum_FN_rf += FN

            # draw a plot contain fgim, rfim, gtim
            # titles Fm_bef, Fm, 1
            # images are gray scale

            imageio.imwrite(
                visual_output + vid_cat + "/" + vid_name + "/" +
                "%d_random_%.4f.png" % (i, Fm), refinemask)

            # if save_fig:
            #     save_figures(Fm, Fm_bef, fgim, rfim, gtim, i, vid_cat,
            #                  vid_name, f)

            #            endtime = time.time()
            #            print("evaluation time:", endtime - starttime)

            print("\n---------------------------------------------\n")
            Re_sum = sum_TP / max((sum_TP + sum_FN), 1)
            Pr_sum = sum_TP / max((sum_TP + sum_FP), 1)

            Fm_sum = (2 * Pr_sum * Re_sum) / max((Pr_sum + Re_sum), 0.0001)

            print("accumulate original:")
            print("Re_sum:", Re_sum)
            print("Pr_sum:", Pr_sum)
            print("Fm_sum:", Fm_sum)

            Re_sum_rf = sum_TP_rf / max((sum_TP_rf + sum_FN_rf), 1)
            Pr_sum_rf = sum_TP_rf / max((sum_TP_rf + sum_FP_rf), 1)

            Fm_sum_rf = (2 * Pr_sum_rf * Re_sum_rf) / max(
                (Pr_sum_rf + Re_sum_rf), 0.0001)

            endtime = time.time()

            print("\n accumulate refinefg:")
            print("Re_sum_rf:", Re_sum_rf)
            print("Pr_sum_rf:", Pr_sum_rf)
            print("Fm_sum_rf:", Fm_sum_rf)

            print(
                "=========================================================================================="
            )
            print("total time:", endtime - starttime)
            print("\n\n")

            # write pa_im, i, Re, Pr, Fm, Re_sum, Pr_sum, Fm_sum, Re_sum_rf, Pr_sum_rf, Fm_sum_rf, endtime - starttime to f

            f.write("pa_im: %s\n" % pa_im)
            f.write("frames: %d\n" % i)
            f.write("current original:\n")
            f.write("Re: %f\n" % Re_bef)
            f.write("Pr: %f\n" % Pr_bef)
            f.write("Fm: %f\n" % Fm_bef)
            f.write("\n current refinefg:\n")
            f.write("Re: %f\n" % Re)
            f.write("Pr: %f\n" % Pr)
            f.write("Fm: %f\n" % Fm)
            f.write("\n---------------------------------------------\n")
            f.write("accumulate original:\n")
            f.write("Re_sum: %f\n" % Re_sum)
            f.write("Pr_sum: %f\n" % Pr_sum)
            f.write("Fm_sum: %f\n" % Fm_sum)
            f.write("\n accumulate refinefg:\n")
            f.write("Re_sum_rf: %f\n" % Re_sum_rf)
            f.write("Pr_sum_rf: %f\n" % Pr_sum_rf)
            f.write("Fm_sum_rf: %f\n" % Fm_sum_rf)
            f.write("total time: %f\n" % (endtime - starttime))
            f.write("\n\n")

    return Re_sum, Pr_sum, Fm_sum, Re_sum_rf, Pr_sum_rf, Fm_sum_rf


def evaluateRefinement(pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, radius,
                       network, device, len_batch, list_train, flag_in, f,
                       vid_cat, vid_name, save_fig, visual_output):

    sum_TP = 0
    sum_FP = 0
    sum_TN = 0
    sum_FN = 0

    sum_TP_rf = 0
    sum_FP_rf = 0
    sum_TN_rf = 0
    sum_FN_rf = 0

    fs, fullfs = loadFiles_plus(pa_gt, ft_gt)
    frames = len(fullfs)

    plt.figure(figsize=(4, 4))
    for i in range(frames):

        starttime = time.time()

        mask_gt = torch.tensor(imageio.imread(fullfs[i]), dtype=torch.float)

        judge_flag = torch.sum(mask_gt == 255) + torch.sum(mask_gt == 0)

        judge_train = checkTrainList(i, list_train)

        if judge_train != 0 and flag_in == 1:
            judge_flag = 0
            print("This frame is included in training set:", i)
            print("list_train:", list_train)

        if judge_flag == 0:
            refinemask = mask_gt
        else:
            #            starttime = time.time()

            refinemask, gt, imXfg, patches_tmp = detectFgImg(
                pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, radius, i, network,
                device, len_batch)
            #            endtime = time.time()

            #            print("network processing time:", endtime - starttime)

            srim = imXfg[:, :, 0:3].int()
            fgim = imXfg[:, :, 3].int()
            rfim = (refinemask * 255).int()
            gtim = gt.int()

            # #         im = src_patches[i, :, :, 0:3]
            # #         beforefg = src_patches[i ,:, :, 3]
            # #         fg = refinemask_train[i].squeeze()
            # #         gt = mask_fountain01[i].squeeze()
            # #
            # #         plt.subplot(1, 4, 1)
            # #         plt.imshow(im)
            # #
            # #         plt.subplot(1, 4, 2)
            # #         plt.imshow(beforefg)
            # #
            # #         plt.subplot(1, 4, 3)
            # #         plt.imshow(fg)
            # #
            # #         plt.subplot(1, 4, 4)
            # #         plt.imshow(gt)
            # #
            # #         plt.pause(0.01)
            # #
            # #         str = input()

            #            starttime = time.time()
            #            starttime1 = time.time()
            #            TP, FP, TN, FN = evaluation_numpy_entry(fgim.numpy(), gtim.numpy())
            #            endtime1 = time.time()

            #            starttime2 = time.time()

            # imshow fgim and gtim

            TP, FP, TN, FN = evaluation_numpy_entry_torch(fgim, gtim)

            #TP, FP, TN, FN = evaluation_numpy_entry(fgim.numpy(), gtim.numpy())

            #            endtime2 = time.time()

            #            print(endtime1 - starttime1)
            #            print(endtime2 - starttime2)

            #            print(TP, FP, TN, FN)
            #            print(TP1, FP1, TN1, FN1)

            Re = TP / max((TP + FN), 1)
            Pr = TP / max((TP + FP), 1)

            Fm = (2 * Pr * Re) / max((Pr + Re), 0.0001)

            Re_bef = Re
            Pr_bef = Pr
            Fm_bef = Fm

            print("\n")
            print(
                "=========================================================================================="
            )
            print("pa_im:", pa_im)
            print(fullfs[i])
            print("frames:", i)
            print("current original:")
            print("Re:", Re)
            print("Pr:", Pr)
            print("Fm:", Fm)

            sum_TP += TP
            sum_FP += FP
            sum_TN += TN
            sum_FN += FN

            TP, FP, TN, FN = evaluation_numpy_entry_torch(rfim, gtim)

            #TP, FP, TN, FN = evaluation_numpy_entry(rfim.numpy(), gtim.numpy())

            Re = TP / max((TP + FN), 1)
            Pr = TP / max((TP + FP), 1)

            Fm = (2 * Pr * Re) / max((Pr + Re), 0.0001)

            print("\n current refinefg:")
            print("Re:", Re)
            print("Pr:", Pr)
            print("Fm:", Fm)

            sum_TP_rf += TP
            sum_FP_rf += FP
            sum_TN_rf += TN
            sum_FN_rf += FN

            # draw a plot contain fgim, rfim, gtim
            # titles Fm_bef, Fm, 1
            # images are gray scale

            if save_fig:
                save_figures(Fm, Fm_bef, fgim, rfim, gtim, i, vid_cat,
                             vid_name, visual_output,f)

            #            endtime = time.time()
            #            print("evaluation time:", endtime - starttime)

            print("\n---------------------------------------------\n")
            Re_sum = sum_TP / max((sum_TP + sum_FN), 1)
            Pr_sum = sum_TP / max((sum_TP + sum_FP), 1)

            Fm_sum = (2 * Pr_sum * Re_sum) / max((Pr_sum + Re_sum), 0.0001)

            print("accumulate original:")
            print("Re_sum:", Re_sum)
            print("Pr_sum:", Pr_sum)
            print("Fm_sum:", Fm_sum)

            Re_sum_rf = sum_TP_rf / max((sum_TP_rf + sum_FN_rf), 1)
            Pr_sum_rf = sum_TP_rf / max((sum_TP_rf + sum_FP_rf), 1)

            Fm_sum_rf = (2 * Pr_sum_rf * Re_sum_rf) / max(
                (Pr_sum_rf + Re_sum_rf), 0.0001)

            endtime = time.time()

            print("\n accumulate refinefg:")
            print("Re_sum_rf:", Re_sum_rf)
            print("Pr_sum_rf:", Pr_sum_rf)
            print("Fm_sum_rf:", Fm_sum_rf)

            print(
                "=========================================================================================="
            )
            print("total time:", endtime - starttime)
            print("\n\n")

            # write pa_im, i, Re, Pr, Fm, Re_sum, Pr_sum, Fm_sum, Re_sum_rf, Pr_sum_rf, Fm_sum_rf, endtime - starttime to f

            f.write("pa_im: %s\n" % pa_im)
            f.write("frames: %d\n" % i)
            f.write("current original:\n")
            f.write("Re: %f\n" % Re_bef)
            f.write("Pr: %f\n" % Pr_bef)
            f.write("Fm: %f\n" % Fm_bef)
            f.write("\n current refinefg:\n")
            f.write("Re: %f\n" % Re)
            f.write("Pr: %f\n" % Pr)
            f.write("Fm: %f\n" % Fm)
            f.write("\n---------------------------------------------\n")
            f.write("accumulate original:\n")
            f.write("Re_sum: %f\n" % Re_sum)
            f.write("Pr_sum: %f\n" % Pr_sum)
            f.write("Fm_sum: %f\n" % Fm_sum)
            f.write("\n accumulate refinefg:\n")
            f.write("Re_sum_rf: %f\n" % Re_sum_rf)
            f.write("Pr_sum_rf: %f\n" % Pr_sum_rf)
            f.write("Fm_sum_rf: %f\n" % Fm_sum_rf)
            f.write("total time: %f\n" % (endtime - starttime))
            f.write("\n\n")

    return Re_sum, Pr_sum, Fm_sum, Re_sum_rf, Pr_sum_rf, Fm_sum_rf


def cut_patches_randomly_updated_test_use(imXfg, img, gt, num_pos, radius):
    '''
    imXfg: (row_im, column_im) tensor
    img: (row_im, column_im, channel) tensor
    gt: (row_im, column_im) tensor
    num_pos: number of patches
    radius: radius of patches
    '''
    # add one more dimension to imXfg and gt
    # imXfg: (row_im, column_im, 1) tensor
    imXfg = imXfg.unsqueeze(0)
    gt = gt.unsqueeze(0)
    # type to float
    imXfg = imXfg.float()
    gt = gt.float()

    # img to (channel, row_im, column_im) tensor
    img = img.permute(2, 0, 1)
    # type to float
    img = img.float()
    # pad the image by tranforms
    pad = transforms.Pad(radius, padding_mode='reflect')
    img_pad = pad(img)
    gt_pad = pad(gt)
    imXfg_pad = pad(imXfg)

    img_pad = img_pad.int().cuda()
    gt_pad = gt_pad.int().cuda()
    imXfg_pad = imXfg_pad.int().cuda()

    # remove first dimension for gt_pad and imXfg_pad
    gt_pad = gt_pad.squeeze(0)
    imXfg_pad = imXfg_pad.squeeze(0)

    # choose center start from radius to row_im + radius - 1
    # choose center start from radius to column_im + radius - 1
    row_im = img.shape[1]
    column_im = img.shape[2]
    center_x_start = radius
    center_x_end = row_im + radius - 1
    center_y_start = radius
    center_y_end = column_im + radius - 1

    img_patch_bags = torch.zeros((num_pos, 3, radius*2, radius*2)).cuda()
    gt_patch_bags = torch.zeros((num_pos, radius*2, radius*2)).cuda()
    imXfg_patch_bags = torch.zeros((num_pos, radius*2, radius*2)).cuda()

    original_centers = []

    for i in range(num_pos):
        # choose center randomly
        center_x = random.randint(center_x_start, center_x_end)
        center_y = random.randint(center_y_start, center_y_end)
        # get the patch
        # left, upper, right, lower
        left = center_y - radius
        upper = center_x - radius
        right = center_y + radius
        lower = center_x + radius
        img_patch = img_pad[:, upper:lower, left:right]
        gt_patch = gt_pad[upper:lower, left:right]
        imXfg_patch = imXfg_pad[upper:lower, left:right]
        # send to the bag
        img_patch_bags[i] = img_patch
        gt_patch_bags[i] = gt_patch
        imXfg_patch_bags[i] = imXfg_patch

        # img_patch_bags.append(img_patch)
        # gt_patch_bags.append(gt_patch)
        # imXfg_patch_bags.append(imXfg_patch)
        # find original center_x and center_y
        center_x_ori = center_x - radius
        center_y_ori = center_y - radius
        original_centers.append([center_x_ori, center_y_ori])
    # concat the bags
    # img_patch_bags = torch.stack(img_patch_bags)
    # gt_patch_bags = torch.stack(gt_patch_bags)
    # imXfg_patch_bags = torch.stack(imXfg_patch_bags)

    return img_patch_bags, gt_patch_bags, imXfg_patch_bags, original_centers


def cut_patches_randomly_updated(imXfg, img, gt, num_pos, radius):
    '''
    imXfg: (row_im, column_im) tensor
    img: (row_im, column_im, channel) tensor
    gt: (row_im, column_im) tensor
    num_pos: number of patches
    radius: radius of patches
    '''
    # add one more dimension to imXfg and gt
    # imXfg: (row_im, column_im, 1) tensor
    imXfg = imXfg.unsqueeze(0)
    gt = gt.unsqueeze(0)
    # type to float
    imXfg = imXfg.float()
    gt = gt.float()

    # img to (channel, row_im, column_im) tensor
    img = img.permute(2, 0, 1)
    # type to float
    img = img.float()
    # pad the image by tranforms
    pad = transforms.Pad(radius, padding_mode='reflect')
    img_pad = pad(img)
    gt_pad = pad(gt)
    imXfg_pad = pad(imXfg)

    img_pad = img_pad.int()
    gt_pad = gt_pad.int()
    imXfg_pad = imXfg_pad.int()

    # remove first dimension for gt_pad and imXfg_pad
    gt_pad = gt_pad.squeeze(0)
    imXfg_pad = imXfg_pad.squeeze(0)

    # choose center start from radius to row_im + radius - 1
    # choose center start from radius to column_im + radius - 1
    row_im = img.shape[1]
    column_im = img.shape[2]
    center_x_start = radius
    center_x_end = row_im + radius - 1
    center_y_start = radius
    center_y_end = column_im + radius - 1

    img_patch_bags = []
    gt_patch_bags = []
    imXfg_patch_bags = []

    for i in range(num_pos):
        # choose center randomly
        center_x = random.randint(center_x_start, center_x_end)
        center_y = random.randint(center_y_start, center_y_end)
        # get the patch
        # left, upper, right, lower
        left = center_y - radius
        upper = center_x - radius
        right = center_y + radius
        lower = center_x + radius
        img_patch = img_pad[:, upper:lower, left:right]
        gt_patch = gt_pad[upper:lower, left:right]
        imXfg_patch = imXfg_pad[upper:lower, left:right]
        # send to the bag
        img_patch_bags.append(img_patch)
        gt_patch_bags.append(gt_patch)
        imXfg_patch_bags.append(imXfg_patch)
    # concat the bags
    img_patch_bags = torch.stack(img_patch_bags)
    gt_patch_bags = torch.stack(gt_patch_bags)
    imXfg_patch_bags = torch.stack(imXfg_patch_bags)
    return img_patch_bags, gt_patch_bags, imXfg_patch_bags


def cut_patches_one_video(dataset,
                          im_path,
                          fg_path,
                          gt_path,
                          frame_list,
                          radius=[16, 32, 64],
                          patch_num=[128 * 4 * 4, 128 * 4, 128]):
    '''
    im_path: path of images
    fg_path: path of foreground masks
    gt_path: path of ground truth masks
    frame_list: list of frame number
    radius: list of radius
    '''
    img_patches_bags_16 = []
    gt_patches_bags_16 = []
    imXfg_patches_bags_16 = []

    img_patches_bags_32 = []
    gt_patches_bags_32 = []
    imXfg_patches_bags_32 = []

    img_patches_bags_64 = []
    gt_patches_bags_64 = []
    imXfg_patches_bags_64 = []

    # prevent folder in the path
    fg_list = [f for f in os.listdir(fg_path) if os.path.isfile(fg_path + f)]
    im_list = [f for f in os.listdir(im_path) if os.path.isfile(im_path + f)]
    gt_list = [f for f in os.listdir(gt_path) if os.path.isfile(gt_path + f)]

    fg_list.sort()
    im_list.sort()
    gt_list.sort()

    # get path for input and gt
    for frame in frame_list:
        # get input and gt, load by imageio
        if dataset == 'CDNet2014':
            imXfg = imageio.imread(fg_path + 'gt%06d.png' % frame)
            img = imageio.imread(im_path + 'in%06d.jpg' % frame)
            gt = imageio.imread(gt_path + 'gt%06d.png' % frame)
        else:
            imXfg = imageio.imread(fg_path + fg_list[frame])
            img = imageio.imread(im_path + im_list[frame])
            gt = imageio.imread(gt_path + gt_list[frame])
        # elif dataset == 'LASIESTA':
        #     imXfg = imageio.imread(fg_path + fg_list[frame])
        #     img = imageio.imread(im_path + im_list[frame])
        #     gt = imageio.imread(gt_path + gt_list[frame])
        # elif dataset == 'SBI2015':
        #     imXfg = imageio.imread(fg_path + 'gt%06d.png' % frame)
        #     img = imageio.imread(im_path + 'in%06d.png' % frame)
        #     gt = imageio.imread(gt_path + 'gt%06d.png' % frame)
        # elif dataset == 'BMC':
        #     imXfg = imageio.imread(fg_path + fg_list[frame])
        #     img = imageio.imread(im_path + im_list[frame])
        #     gt = imageio.imread(gt_path + gt_list[frame])
        # convert to tensor
        imXfg = torch.from_numpy(imXfg)
        img = torch.from_numpy(img)
        gt = torch.from_numpy(gt)
        # cut patches
        for radi, patch in zip(radius, patch_num):
            img_patches, gt_patches, imXfg_patches = cut_patches_randomly_updated(
                imXfg, img, gt, patch, radi)
            if radi == 16:
                img_patches_bags_16.append(img_patches)
                gt_patches_bags_16.append(gt_patches)
                imXfg_patches_bags_16.append(imXfg_patches)
            elif radi == 32:
                img_patches_bags_32.append(img_patches)
                gt_patches_bags_32.append(gt_patches)
                imXfg_patches_bags_32.append(imXfg_patches)
            elif radi == 64:
                img_patches_bags_64.append(img_patches)
                gt_patches_bags_64.append(gt_patches)
                imXfg_patches_bags_64.append(imXfg_patches)

    # concat the bags
    img_patches_bags_16 = torch.cat(img_patches_bags_16)
    gt_patches_bags_16 = torch.cat(gt_patches_bags_16)
    imXfg_patches_bags_16 = torch.cat(imXfg_patches_bags_16)

    img_patches_bags_32 = torch.cat(img_patches_bags_32)
    gt_patches_bags_32 = torch.cat(gt_patches_bags_32)
    imXfg_patches_bags_32 = torch.cat(imXfg_patches_bags_32)

    img_patches_bags_64 = torch.cat(img_patches_bags_64)
    gt_patches_bags_64 = torch.cat(gt_patches_bags_64)
    imXfg_patches_bags_64 = torch.cat(imXfg_patches_bags_64)

    # concat img and imXfg make it 2 channel
    # extend one dimension for imXfg
    imXfg_patches_bags_16 = imXfg_patches_bags_16.unsqueeze(1)
    imXfg_patches_bags_32 = imXfg_patches_bags_32.unsqueeze(1)
    imXfg_patches_bags_64 = imXfg_patches_bags_64.unsqueeze(1)

    img_patches_bags_16 = torch.cat(
        (img_patches_bags_16, imXfg_patches_bags_16), 1)
    img_patches_bags_32 = torch.cat(
        (img_patches_bags_32, imXfg_patches_bags_32), 1)
    img_patches_bags_64 = torch.cat(
        (img_patches_bags_64, imXfg_patches_bags_64), 1)

    img_all_patches = [
        img_patches_bags_16, img_patches_bags_32, img_patches_bags_64
    ]
    gt_all_patches = [
        gt_patches_bags_16, gt_patches_bags_32, gt_patches_bags_64
    ]

    return img_all_patches, gt_all_patches


def load_one_video(dataset, name_categroy, name_video, idx_list, radius):
    print("loading video %s" % name_video)
    if dataset == 'CDNet2014':
        dataset_path = "/home/guanfang/Dataset/CDNet2014_full/"
        fg_path = "/home/guanfang/Dataset/full_100_0.5_cdnet_fgimg/"
        str_pa_im = dataset_path + name_categroy + '/' + name_video + '/input/'
        str_pa_fg = fg_path + name_categroy + '/' + name_video + '/'
        str_pa_gt = dataset_path + name_categroy + '/' + name_video + '/groundtruth/'

    elif dataset == 'LASIESTA':
        dataset_path = "/home/guanfang/Dataset/LASIESTA/"
        fg_path = "/home/guanfang/Dataset/full_100_0.5_LASIESTA_seen_fgimg/"
        str_pa_im = dataset_path + name_categroy + '/' + name_video + '/' + name_video + '/'
        str_pa_fg = fg_path + name_categroy + '/' + name_video + '/'
        str_pa_gt = dataset_path + name_categroy + '/' + name_video + '/' + name_video + '-GT/'
    
    elif dataset == 'SBI2015':
        dataset_path = "/home/gfdong/SBI2015/"
        fg_path = "/home/gfdong/SBI2015/"
        str_pa_im = dataset_path + name_video + '/input/'
        str_pa_fg = fg_path + name_video + '/fgimgs_unseen/'
        str_pa_gt = dataset_path + name_video + '/groundtruth/'
    
    elif dataset == 'BMC':
        dataset_path = "/home/gfdong/BMC/"
        fg_path = "/home/gfdong/BMC/"
        str_pa_im = dataset_path + name_video + '/input/'
        str_pa_fg = fg_path + name_video + '/fgimgs_unseen/'
        str_pa_gt = dataset_path + name_video + '/groundtruth/'
    

    pa_im_list = [str_pa_im]
    pa_fg_list = [str_pa_fg]
    pa_gt_list = [str_pa_gt]


    #pa_gt_list = [  '/home/cqzhao/dataset/dataset2014/dataset/nightVideos/winterStreet/groundtruth/' ]

    #    idx_list = [ [1242,1047,1313,1315,924,1316,1161,1272,1010,1314,963,900,954,1009,1064,1119,1174,1229,1284,1339]]

    # python start from 0
    # t = 0
    # for l in idx_list:
    #     l[:] = [i - 1 for i in l]
    #     idx_list[t] = l
    #     t += 1

    ft_im = 'jpg'
    ft_fg = 'png'
    ft_gt = 'png'

    num_pos = 10
    num_neg = 10

    base_num = 64
    '''
    print("pa_im_list: ",
          pa_im_list,
          "pa_fg_list: ",
          pa_fg_list,
          "pa_gt_list: ",
          pa_gt_list,
          "idx_list: ",
          idx_list,
          "ft_im: ",
          ft_im,
          "ft_fg: ",
          ft_fg,
          "ft_gt: ",
          ft_gt,
          "num_pos: ",
          num_pos,
          "num_neg: ",
          num_neg,
          sep='\n')
    '''
    img_all_patches, gt_all_patches = cut_patches_one_video(
        dataset,
        pa_im_list[0],
        pa_fg_list[0],
        pa_gt_list[0],
        idx_list[0],
        patch_num=[base_num * 4 * 4, base_num * 4, base_num])
    # patches, gt = videos2patches(pa_im_list, pa_fg_list, pa_gt_list, idx_list,
    #                              ft_im, ft_fg, ft_gt, radius, num_pos, num_neg)
    '''
    print("*" * 50)
    print("patches.shape: ", patches.shape, "gt.shape: ", gt.shape, sep='\n')
    print("*" * 50)
    print("pa_im_list: ",
          pa_im_list,
          "pa_fg_list: ",
          pa_fg_list,
          "pa_gt_list: ",
          pa_gt_list,
          "idx_list: ",
          idx_list,
          "ft_im: ",
          ft_im,
          "ft_fg: ",
          ft_fg,
          "ft_gt: ",
          ft_gt,
          "radius: ",
          radius,
          sep='\n')
    '''
    # patches_seed, gt_seed = videos2patches_seedfill(pa_im_list, pa_fg_list,
    #                                                 pa_gt_list, idx_list,
    #                                                 ft_im, ft_fg, ft_gt,
    #                                                 2 * radius)

    # patches = torch.cat((patches, patches_seed), dim=0)
    # gt = torch.cat((gt, gt_seed), dim=0)

    # idx = gt < 250
    # gt[idx] = 0
    for i in range(len(img_all_patches)):
        candidates = gt_all_patches[i]
        idx = []
        for j, patch in enumerate(candidates):
            # j is a patch, if one of the pixels are not 0 or 255, then remove it
            unique_value = torch.unique(patch)
            if unique_value.shape[0] == 1 and unique_value[0] not in [0,255]:
                continue
            idx.append(j)
        idx = torch.tensor(idx)
        img_all_patches[i] = img_all_patches[i][idx]
        gt_all_patches[i] = gt_all_patches[i][idx]
        



    for i in range(len(img_all_patches)):
        img_all_patches[i] = img_all_patches[i].float()
        gt_all_patches[i] = gt_all_patches[i].float()

        # print(img_all_patches[i].shape)
        img_all_patches[i] = img_all_patches[i] / 255.0
        gt_all_patches[i] = gt_all_patches[i] / 255.0

        # patches = patches.float() / 255.0
        # gt = gt.float() / 255.0

        # patches = patches.permute(0, 3, 1, 2)
        # gt = gt.permute(0, 3, 1, 2)

    # print("\n\n")
    # print("Data format:  ------------------------------------------")
    # print("")
    # print("processing data: ", name_categroy, name_video, sep=' ')
    # print(patches.shape)
    # print(gt.shape)
    # print("")
    # print("--------------------------------------------------------")
    # print("\n\n")
    return img_all_patches, gt_all_patches, pa_im_list, pa_fg_list, pa_gt_list


def main(args):

    name_categroies = args.datatypes
    name_videos = args.videos
    idx_lists = args.index_list
    len_idx_list = args.len_index_list
    net_type = args.net_type
    epoch = args.epoch
    cuda = args.cuda
    train = args.is_train
    dataset_list = args.test_dataset
    test_layer = args.test_layer
    test_layer = int(test_layer)
    network_input = args.network_input
    visual_output = args.visual_output
    txt_output = args.txt_output
    report_output = args.report_output

    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    train = False
    test_random = True
    save_fig = True

    ft_im = 'jpg'
    ft_fg = 'png'
    ft_gt = 'png'

    radius = 32

    for i in len_idx_list:
        i = int(i)
        sliced = idx_lists[0:i]
        sliced = [int(j) for j in sliced]
        idx_lists.append([sliced])
        del idx_lists[0:i]

    # print("name_categroies: ", name_categroies)
    # print("name_videos: ", name_videos)
    # print("idx_lists: ", idx_lists)

    # print Category, video, index one by one
    for i in range(len(name_categroies)):
        print(dataset_list[i], name_categroies[i], name_videos[i], idx_lists[i])

    setupSeed(0)

    # for loop load data
    pa_im_list_bag, pa_fg_list_bag, pa_gt_list_bag = [], [], []
    patches = None
    gt = None
    for i in range(len(name_categroies)):
        dataset = dataset_list[i]
        name_categroy = name_categroies[i]
        name_video = name_videos[i]
        idx_list = idx_lists[i]

        current_patches, current_gt, pa_im_list, pa_fg_list, pa_gt_list = load_one_video(
            dataset, name_categroy, name_video, idx_list, radius)

        if patches == None:
            patches = current_patches
            gt = current_gt
        else:
            for j in range(len(current_patches)):
                patches[j] = torch.cat((patches[j], current_patches[j]), dim=0)
                gt[j] = torch.cat((gt[j], current_gt[j]), dim=0)

        del current_patches
        del current_gt

        # print shape
        print("patches.shape: ", patches[0].shape, "gt.shape: ", gt[0].shape)
        print("patches.shape: ", patches[1].shape, "gt.shape: ", gt[1].shape)
        print("patches.shape: ", patches[2].shape, "gt.shape: ", gt[2].shape)
        print("current i: ", i)

        pa_im_list_bag.append(pa_im_list)
        pa_fg_list_bag.append(pa_fg_list)
        pa_gt_list_bag.append(pa_gt_list)

    # concat patches and gt
    print("*" * 50)
    # patches = torch.cat(patches_bag, dim=0)
    # gt = torch.cat(gt_bag, dim=0)

    # shuffle patches and gt
    for i in range(len(patches)):
        idx = torch.randperm(patches[i].shape[0])
        patches[i] = patches[i][idx]
        gt[i] = gt[i][idx]

    print(pa_im_list_bag)
    print(pa_fg_list_bag)
    print(pa_gt_list_bag)
    print("*" * 50)
    # training the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init the network by the type
    networks = [ONet_TINY, U_Net, ONet_TINY_UPPER, ONet_BASE]
    network_names = ["tiny_o_net", "u_net", "tiny_o_net_upper", "base_o_net"]
    is_train_o_net = False
    if net_type == "base_o_net":
        is_train_o_net = True
    network = networks[network_names.index(net_type)](n_channels=4,
                                                      n_classes=2,
                                                      bilinear=True)
    network = torch.nn.DataParallel(network)
    network.to(device)

    optimizer = optim.RMSprop(network.parameters(),
                              lr=0.00001,
                              weight_decay=1e-8,
                              momentum=0.9)
    #    optimizer = optim.Adam(network.parameters(), lr = 0.001)

    #    loss_func = nn.BCEWithLogitsLoss()

    class_weights = torch.FloatTensor([0.2, 0.8]).to(device)
    loss_func = nn.CrossEntropyLoss(weight=class_weights,
                                    reduction='sum').to(device)

    len_batch = 128 * 4
    num_epoch = epoch

    # net_pa = '../../data_refinemet/network_' + name_categroy + '_' + name_video + '/'

    net_pa = './network_reproduce_one_model/'
    #    fluidHighway/'

    if train:
        network = train_unet(patches,
                             gt,
                             device,
                             network,
                             optimizer,
                             loss_func,
                             len_batch,
                             num_epoch,
                             net_pa,
                             train_o_net=is_train_o_net)
        # save the network
        torch.save(network.state_dict(), net_pa + 'network_dis_final.pt')

    else:
        network.load_state_dict(torch.load(network_input))

    # stop use multigpu
    # network = network.module
    # # save network to onnx
    # network.eval()
    # torch.onnx.export(network, torch.randn(1, 4, 64, 64).cuda(), "network.onnx", verbose=True)
    # # use multigpu
    # network = torch.nn.DataParallel(network)

    #    print("train_unet ================================")

    #     strcommand = input()
    #
    #    name_net = '../../data_refinemet/network_5/network_dis_0140.pt'

    #     name_net = '../../data_refinemet/network_8/network_dis_0580.pt'
    #     network.load_state_dict(torch.load(name_net))
    #     network = network.to(device)
    #
    #     name_net = '../../data_refinemet/network_9/network_dis_0580.pt'
    #     network.load_state_dict(torch.load(name_net))
    #     network = network.to(device)

    # testing the network
    # output = test_unet(patches, device, network, len_batch)
    # fgimgs = output.argmax(dim=1, keepdim=True).cpu().detach().squeeze()
    # gtimgs = gt.cpu().detach().squeeze()
    # fginput = patches[:, 3, :, :].squeeze()
    # srcimgs = patches[:, 0:3, :, :].squeeze()

    # fginput = fginput * 255
    # fgimgs = fgimgs * 255
    # gtimgs = gtimgs * 255

    #    print(fgimgs.shape)
    #    print(gt.shape)

    # fig = plt.figure(figsize=(4, 2))

    # frames = fgimgs.shape[0]

    # #    print("patches.shape", patches.shape)

    # sum_TP = 0
    # sum_FP = 0
    # sum_TN = 0
    # sum_FN = 0

    # sum_TP_in = 0
    # sum_FP_in = 0
    # sum_TN_in = 0
    # sum_FN_in = 0

    print("running evaluateRefinement ...")
    # if file named report.txt, delete it
    if not os.path.exists(report_output):
        os.makedirs(report_output)
    if os.path.exists(report_output + "report.txt"):
        os.remove(report_output + "report.txt")
    for i in range(len(pa_im_list_bag)):
        if (dataset_list[i] == "CDNet2014"):
            ft_im, ft_fg, ft_gt = "jpg", "png", "png"
        elif (dataset_list[i] == "LASIESTA"):
            ft_im, ft_fg, ft_gt = "bmp", "png", "png"

        pa_im_list = pa_im_list_bag[i]
        pa_fg_list = pa_fg_list_bag[i]
        pa_gt_list = pa_gt_list_bag[i]
        idx_list = idx_lists[i]

        pa_im = pa_im_list[0]
        pa_fg = pa_fg_list[0]
        pa_gt = pa_gt_list[0]
        list_train = idx_list[0]

        # if results not exist, create it
        if not os.path.exists(txt_output):
            os.makedirs(txt_output)

        # open txt file named name_categroies[i] + name_videos[i]
        f_normal = open(f"{txt_output}{name_categroies[i]}_{name_videos[i]}_normal.txt", "w")
        f_random = open(f"{txt_output}{name_categroies[i]}_{name_videos[i]}_random.txt", "w")

        # Re_sum, Pr_sum, Fm_sum, Re_sum_rf, Pr_sum_rf, Fm_sum_rf = evaluateRefinement(
        #     pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, 2 * radius, network,
        #     device, len_batch, list_train, 1, f_normal, name_categroies[i],
        #     name_videos[i], save_fig, visual_output)
        Re_sum, Pr_sum, Fm_sum, Re_sum_rf, Pr_sum_rf, Fm_sum_rf = 0, 0, 0, 0 ,0 ,0
        try:
            if test_random:
                Re_sum_r, Pr_sum_r, Fm_sum_r, Re_sum_rf_r, Pr_sum_rf_r, Fm_sum_rf_r = evaluateRefinement_random_sample(
                    pa_im, ft_im, pa_fg, ft_fg, pa_gt, ft_gt, 2 * radius, network,
                    device, len_batch, list_train, 1, f_random, name_categroies[i],
                    name_videos[i], save_fig, test_layer, visual_output, dataset_list[i])

            # open a txt file alled report + current date and time
            output_file = open(report_output + "report.txt", "a")
            output_file.write("*" * 100 + "\n")
            # write video type and name
            output_file.write("Video Type: " + name_categroies[i] +
                            " Video Name: " + name_videos[i] + "\n")
            # write the results
            output_file.write("accumulate original:" + "\n")
            output_file.write("Re_sum: " + str(Re_sum) + "\n")
            output_file.write("Pr_sum: " + str(Pr_sum) + "\n")
            output_file.write("Fm_sum: " + str(Fm_sum) + "\n")
            output_file.write("accumulate refinement:" + "\n")
            output_file.write("Re_sum_rf: " + str(Re_sum_rf) + "\n")
            output_file.write("Pr_sum_rf: " + str(Pr_sum_rf) + "\n")
            output_file.write("Fm_sum_rf: " + str(Fm_sum_rf) + "\n")
            if test_random:
                output_file.write("accumulate random sample:" + "\n")
                output_file.write("Re_sum_r: " + str(Re_sum_rf_r) + "\n")
                output_file.write("Pr_sum_r: " + str(Pr_sum_rf_r) + "\n")
                output_file.write("Fm_sum_r: " + str(Fm_sum_rf_r) + "\n")
            output_file.write("*" * 100 + "\n")
            # close the file
            output_file.close()
        except:
            error_files = open("error.txt", "a")
            error_files.write("ERROR!\n")
            error_files.write(name_categroies[i] + "  " + name_videos[i] + "\n")
            error_files.close()
            


if __name__ == '__main__':

    # argc = len(sys.argv)
    # argv = sys.argv
    # python -u train_test.py --old_args _ cameraJitter boulevard v1 1 1988 983 814 830 900 1000 1200 1260 1965 2180 2330 1980 877 2218 1474 1816 2158 2500 --net_type tiny_o_net

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument(
        '--net_type',
        default="tiny_o_net",
        type=str,
        help=
        'choose from tiny_o_net, tiny_o_net_upper, tiny_o_net_lower, base_o_net'
    )

    parser.add_argument('--cuda', type=str, help='cuda number')

    # add argument for epoch
    parser.add_argument('--epoch', default=100, type=int, help='epoch')

    parser.add_argument('--datatypes',
                        nargs='+',
                        required=True,
                        help='datatypes in list format')

    # add parser for videos
    parser.add_argument('--videos',
                        nargs='+',
                        required=True,
                        help='videos in list format')

    # add parser for index_list
    parser.add_argument('--index_list',
                        nargs='+',
                        required=True,
                        help='index_list will be convert to 2d matrix')

    # add parser for length of index_list
    parser.add_argument('--len_index_list',
                        nargs='+',
                        required=True,
                        help='length of index_list')

    # add parser for dataset list
    parser.add_argument('--test_dataset', 
                        nargs='+', 
                        required=True, 
                        help='dataset in list format')

    # add parser for train or test
    parser.add_argument('--is_train', required=True, help='train or test')

    parser.add_argument('--test_layer')

    parser.add_argument('--network_input')
    parser.add_argument('--visual_output')
    parser.add_argument('--txt_output')
    parser.add_argument('--report_output')

    args = parser.parse_args()

    main(args)
