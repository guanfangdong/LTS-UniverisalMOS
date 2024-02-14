import sys
sys.path.append("../../")

import os

import matplotlib.pyplot as plt

import time


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt


import numpy as np

from common_py.dataIO import loadImgs_pytorch
from common_py.evaluationBS import evaluation_numpy
from common_py.evaluationBS import evaluation_numpy_entry
from common_py.evaluationBS import evaluation_numpy_entry_torch
from common_py.dataIO import saveImg
from common_py.dataIO import saveMod

from common_py.dataIO import readImg_byFilesIdx
from common_py.dataIO import readImg_byFilesIdx_pytorch
from common_py.dataIO import getVideoSize

from common_py.dataIO import loadFiles_plus
from common_py.utils import setupSeed


from function.arithdis import PDFs

from function.arithdis import ProdDis
from function.arithdis import SumDis

from function.arithdis import arithmeticDis
from function.arithdis import preProcessPDF
from function.arithdis import preProdDis
from function.arithdis import preSumDis

from function.arithdis import getc_X
from function.arithdis import gaussianpdf
from function.arithdis import batchGauPdf
from function.arithdis import randPdf

from ADNNet_data_plus import *


from bayesian.bayesian import bayesRefine_iterative_gpu

from binarymask.binarymask import getTrainBinMask


# from function.prodis import DifferentiateDis_multi


from params_input.params_input import QParams

import imageio


from torch.autograd import Variable

np.set_printoptions(threshold=np.inf)




class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(channels), 1, 5, 5))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x


def batchConvImgs(imgs, conv = GaussianBlurConv()):

    frames_im, row_im, column_im, byte_im = imgs.shape

    for i in range(frames_im):
        imgs[i] = conv(imgs[i, :, :, :].unsqueeze(dim = 0).permute(0, 3, 1, 2)).squeeze().permute(1, 2, 0)

    return imgs


def getVidHist_plus(imgs, left = -1, right = 1, border = 0.01):

    frame, row, column, byte = imgs.shape

    imgs = imgs.reshape(frame, row*column, byte)
    imgs = imgs.permute(1, 0, 2)


    len_hist = round((right - left)/border) + 1


    hist_data = torch.empty([row*column, len_hist , byte])
    num_hist = round((right - left)/border) + 1


    for i in range(row*column):
        for b in range(byte):
            hist_data[i, :, b] = torch.histc( (imgs[i, :, b]/255.0)*right, num_hist, left, right)/(frame*1.0)


    return hist_data


def getNormalData_byHistVid_files(vid_hist, pa_im, ft_im, pa_gt, ft_gt, curidx, left = -1, right = 1, delta = 0.01):


    frames, row_im, column_im, byte_im = getVideoSize(pa_im, ft_im)

    im = readImg_byFilesIdx_pytorch(curidx, pa_im, ft_im)
    im = batchConvImgs(im.unsqueeze(0)).squeeze()
    im = im.reshape(row_im*column_im, byte_im)

    lb = readImg_byFilesIdx_pytorch(curidx, pa_gt, ft_gt)

    im = (im/255.0)*right
    im = torch.round(im/delta)


    num_hist = round((right - left)/delta) + 1
    num_right = round(right/delta) + 1

    offset_right = round(right/delta)

    hist_data = torch.abs(vid_hist - vid_hist)
    labs_data = lb.reshape(row_im*column_im)

    for i in range(num_right):
        for b in range(byte_im):
            idx_r = im[:, b] == i

            hist_data[idx_r, (num_hist - offset_right - i ):(num_hist - i) , b] = vid_hist[idx_r, (num_hist - offset_right):num_hist, b]

    return hist_data, labs_data




class ClassifyNetwork(nn.Module):
    def __init__(self, dis_num):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 1, (1, 8), stride = 1, bias = False)

        self.fc1 = nn.Linear(202, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, data):
        
        x = self.conv1(data)

        x = x.view(-1, 202)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim = 1)

class PreproNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(9, 18, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(18, 32, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 1, 3, stride=1, padding=1, bias=False)



    def forward(self, data):

        x = data.permute(0, 2, 3, 1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
        
        x = x.squeeze(1)

        return x
        


def detectFg(data_vid, batchsize, device, X, W, B, F_W, F_B, proddis, sumdis, netP, netC):

    re_labs = np.zeros(data_vid.shape[0])


    with torch.no_grad():

        for i in range(round(data_vid.shape[0]/batchsize + 0.499999999999999999)):

            data = data_vid[i*batchsize:(i + 1)*batchsize].to(device, dtype = torch.float32)


            X._f = netP(data)

            output_W = arithmeticDis(X, W, F_W, proddis)
            output_B = arithmeticDis(X, B, F_B, sumdis)

            output_DIS = (output_W._f + output_B._f)
            output_labs = netC(output_DIS)

            
            re_labs[i*batchsize:(i + 1)*batchsize] = output_labs.argmax(dim = 1, keepdim = True).cpu().detach().squeeze()


    return re_labs 



def main(argc, argv):

    setupSeed(999)

    qparams = QParams()
    qparams.setParams(argc, argv)

    gpuid   = qparams['gpuid']
    pa_net  = qparams['pa_net']
    idx_net = qparams['idx_net']


    use_cuda = torch.cuda.is_available()


    print("------------")
    print(use_cuda)
    print("------------")


    device = torch.device(("cuda:" + str(gpuid)) if use_cuda else "cpu")



    batchsize = 1000
    batchsize_detect = 1000

    # num_epoch = qparams['epochnum']
    # num_epoch = 200

    left_data = -1
    right_data = 1

    left = -1
    right = 1
    delta = 0.01
    lr_rate = 0.0001
    num_dis = 8

    delta = 0.01
    
    chls = 3



    # ######################################################################################
    #                                                                       ################
    # initialize the distribution variables                                         ########
    X = PDFs().emptyPDFs(7, chls, left, right - delta, delta)

    W = PDFs().emptyPDFs(num_dis, chls, left, right, delta)
    B = PDFs().emptyPDFs(num_dis, chls, left, right, delta)

    F_W = PDFs().emptyPDFs(num_dis, chls, left, right, delta)
    F_B = PDFs().emptyPDFs(num_dis, chls, left, right, delta)

    proddis = ProdDis.apply
    sumdis = SumDis.apply

    # ######################################################################################


    X = X.to(device)

    W = W.to(device)
    B = B.to(device)

    F_W = F_W.to(device)
    F_B = F_B.to(device)


    preProcessPDF(X)
    preProcessPDF(W)
    preProcessPDF(B)
    preProcessPDF(F_W)
    preProcessPDF(F_B)

    preProdDis(X, W, F_W)
    preSumDis(X, B, F_B)



    W._f = Variable(W._f, requires_grad = True)
    B._f = Variable(B._f, requires_grad = True)

    netC = ClassifyNetwork(num_dis).to(device)
    netP = PreproNetwork().to(device)


    name_netC = pa_net + 'netC_' + str(idx_net).zfill(4) + '.pt'
    name_netP = pa_net + 'netP_' + str(idx_net).zfill(4) + '.pt'

    name_f_W = pa_net + 'f_W_' + str(idx_net).zfill(4) + '.pt'
    name_f_B = pa_net + 'f_B_' + str(idx_net).zfill(4) + '.pt'

    netC.load_state_dict(torch.load(name_netC, map_location = device))
    netP.load_state_dict(torch.load(name_netP, map_location = device))

    W._f = torch.load(name_f_W, map_location = device)
    B._f = torch.load(name_f_B, map_location = device)



    print("loading test data")
    ft_im = 'jpg'
    ft_gt = 'png'

    ft_im = qparams['ft_im']
    ft_gt = qparams['ft_gt']


    pa_im = qparams['pa_im']
    pa_gt = qparams['pa_gt']
    pa_out = qparams['pa_out']



    imgs = loadImgs_pytorch(pa_im, ft_im)
    print(imgs.shape)
    if len(imgs.shape) == 3:
    # imgs shape is tensor(14, 228, 308)
    # convert to tensor(14, 228, 308, 3)
        imgs = imgs.unsqueeze(3).repeat(1, 1, 1, 3)
    imgs = gaussianSmooth(imgs) 
    eximgs = videoPadding(imgs, radius=0)
    
    data = eximgs.clone()/255.0
    c_X, hists_tensor_sp = tensor2hist(data, dim=0)


    print("completed")

    frames, row_im, column_im, byte_im = imgs.shape
    frames_im, row_im, column_im, byte_im = imgs.shape



    print("remove temporal imgs for more memory")
    del imgs
    print("completed")



    fs, fullfs = loadFiles_plus(pa_gt, ft_gt)


    TP_sum = 0
    FP_sum = 0
    TN_sum = 0
    FN_sum = 0


    for frame_idx in range(frames):
            
        check_filename = pa_out + fs[frame_idx]

        labs_tru = readImg_byFilesIdx_pytorch(frame_idx, pa_gt, ft_gt)
        print(labs_tru.shape)



        judge_flag = torch.sum(labs_tru == 255) + torch.sum(labs_tru == 0)

        if judge_flag == 0:

            im_fg = torch.abs(labs_tru - labs_tru).detach().cpu().numpy()

            print("empty groundtruth frame: frame_idx = ", frame_idx)

        else:
            if os.path.exists(check_filename):
                im_fg = imageio.imread(check_filename)

            else:
                print("generating histogram")
                starttime = time.time()

                data = eximgs[frame_idx]/255.0
                subhist_sp = getSubHists_byHist(data, hists_tensor_sp)

                
                size = subhist_sp.shape
                subhist_sp = subhist_sp.reshape(size[0], size[1]*size[2], size[3]*size[4], size[5])
                subhist_sp = subhist_sp.permute(1, 0, 2, 3)
                endtime = time.time()
                print("completed, time:", endtime - starttime)

                
                print("")
                print("detecting fg")
                stime = time.time()
                re_labs = detectFg(subhist_sp, batchsize_detect, device, X, W, B, F_W, F_B, proddis, sumdis, netP, netC)
                
                re_labs = re_labs/2

                im_fg = np.round(np.reshape(re_labs, (row_im, column_im))*255)
                etime = time.time()
                print("completed, time:", etime - stime)



        # labs_tru = readImg_byFilesIdx(frame_idx, pa_gt, ft_gt)
        # gt_fg = np.round(labs_tru)

        # TP, FP, TN, FN = evaluation_numpy_entry_torch(torch.tensor(im_fg), torch.tensor(gt_fg))


        # Re = TP/max((TP + FN), 1)
        # Pr = TP/max((TP + FP), 1)

        # Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)



        # TP_sum = TP_sum + TP
        # FP_sum = FP_sum + FP
        # TN_sum = TN_sum + TN
        # FN_sum = FN_sum + FN

        # Re_acc = TP_sum/max((TP_sum + FN_sum), 1)
        # Pr_acc = TP_sum/max((TP_sum + FP_sum), 1)

        # Fm_acc = (2*Pr_acc*Re_acc)/max((Pr_acc + Re_acc), 0.0001)


        filename = pa_out + fs[frame_idx]

        print("")

        print("borderline *-------------------------------------------*")

        print("current files:", fs[frame_idx])

        print("savimg fgim:", filename)
        saveImg(filename, im_fg.astype(np.uint8))


        # print("current Re:", Re)
        # print("current Pr:", Pr)
        # print("current Fm:", Fm)
        # print("accumulated Re:", Re_acc)
        # print("accumulated Pr:", Pr_acc)
        # print("accumulated Fm:", Fm_acc)

        print("borderline *-------------------------------------------*")

        print("")










if __name__ == '__main__':

    argc = len(sys.argv)
    argv = sys.argv

    main(argc, argv)
