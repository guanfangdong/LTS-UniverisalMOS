import sys

sys.path.append('/home/cqzhao/projects/matrix/')
sys.path.append('../../../')

import os


import time

import torch
import torch.nn as nn


import imageio
import matplotlib.pyplot as plt
import torch.nn.functional as F

from common_py.dataIO import loadFiles_plus




def getTrainBinMask(mask, radius = 1):


    row_im, column_im = mask.shape

    ex_mask = F.pad(mask.unsqueeze(0).unsqueeze(0), (radius, radius, radius, radius), mode = 'replicate').squeeze()
    mask_vec = torch.cat([ex_mask[i:i + row_im, j:j + column_im].reshape(row_im*column_im, 1) for i in range(radius*2 + 1) for j in range(radius*2 + 1)], dim = 1)


    idx = mask_vec == 255
    mask_vec[idx] = 1
    mask_vec[~idx] = 0
    mask_flag = torch.sum(mask_vec, dim = 1)

    mask_edge = mask.reshape(row_im*column_im, 1).clone()


    idx_edge = (mask_edge == 170).squeeze()
    idx_flag = (mask_flag > 0).squeeze()


    idx = (idx_edge & idx_flag).squeeze()

    mask_edge[idx] = 255


    return mask_edge.reshape(row_im, column_im).squeeze()




def main(argc, argv):

    print("hello world")

    pa_im = '/home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/'
    pa_im = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input/'
    ft_im = 'jpg'

    pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/baseline/highway/groundtruth/'
    pa_gt = '/home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth/'
    ft_gt = 'png'

    fs, fullfs = loadFiles_plus(pa_gt, ft_gt)

    im = torch.tensor( imageio.imread(fullfs[1148]), dtype = torch.float )

    starttime = time.time()
    rim = getTrainBinMask(im, 3)
    endtime = time.time()

    runtime = endtime - starttime

    print("runtime:", runtime)

    fig = plt.figure(figsize = (8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(im.cpu().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(rim.cpu().numpy())

    plt.show()




if __name__ == '__main__':

    argc = len(sys.argv)
    argv = sys.argv

    main(argc, argv)

