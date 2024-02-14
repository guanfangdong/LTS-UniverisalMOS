import sys
sys.path.append("/home/cqzhao/projects/matrix/")
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


from bayesian.bayesian import bayesRefine_iterative_gpu

from binarymask.binarymask import getTrainBinMask

#from function.prodis import DifferentiateDis_multi


from params_input.params_input import QParams

import imageio


from torch.autograd import Variable

np.set_printoptions(threshold=np.inf)



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

    re_labs = torch.zeros(data_vid.shape[0])


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
       


def processLabs(labs):

    
    re_labs = labs.long()

    # 暂时不考虑影子的lab,之后再加
    # background labels
    idx = re_labs == 0
    re_labs[idx] = 0

    # foreground labels
    idx = re_labs == 255
    re_labs[idx] = 2

    
    # others
    idx = re_labs > 2
    re_labs[idx] = 1
    

    return re_labs




def main(argc, argv):


    setupSeed(999)

    print("torch version:", torch.__version__)
    print("")

    qparams = QParams()
    qparams.setParams(argc, argv)

    gpuid = qparams['gpuid']
    net_pa = qparams['pa_out']
    preload = qparams['preload']

    print("gpuid: ", gpuid)

    use_cuda = torch.cuda.is_available()

    print("use_cuda:", use_cuda)

    torch.manual_seed(0)


    device = torch.device(("cuda:" + str(gpuid)) if use_cuda else "cpu")

    params = {'zero_swap': True, 'zero_approx': True, 'normal': False}



    print("loading training data")
    print("number of training sets:", len(qparams['train_data']))

    fs_ht, fullfs_ht = loadFiles_plus(qparams['train_data'][0], 'hist')
    fs_lb, fullfs_lb = loadFiles_plus(qparams['train_data'][0], 'labs')

    print("loading training data from:", qparams['train_data'][0])
    print("load data files:", fullfs_ht[0])

    data_vid = torch.load(fullfs_ht[0]).cpu().type(torch.float16)
    labs_vid = torch.load(fullfs_lb[0]).cpu().type(torch.float16)

    for i in range(1, len(fullfs_lb)):
        print("load data files:", fullfs_ht[i])
        data_vid_temp = torch.load(fullfs_ht[i]).cpu().type(torch.float16)
        labs_vid_temp = torch.load(fullfs_lb[i]).cpu().type(torch.float16)

        data_vid = torch.cat((data_vid, data_vid_temp), dim = 0)
        labs_vid = torch.cat((labs_vid, labs_vid_temp), dim = 0)

        print("total data size:", data_vid.shape)
        print("total labs size:", labs_vid.shape)


        del data_vid_temp
        del labs_vid_temp



    num_data = len(qparams['train_data'])

    cnt = 1

    while cnt < num_data:

        print("loading training data from:", qparams['train_data'][cnt])

        fs_ht, fullfs_ht = loadFiles_plus(qparams['train_data'][cnt], 'hist')
        fs_lb, fullfs_lb = loadFiles_plus(qparams['train_data'][cnt], 'labs')


        for i in range(len(fullfs_lb)):
            print("load data files:", fullfs_ht[i])
            data_vid_temp = torch.load(fullfs_ht[i]).cpu().type(torch.float16)
            labs_vid_temp = torch.load(fullfs_lb[i]).cpu().type(torch.float16)

            data_vid = torch.cat((data_vid, data_vid_temp), dim = 0)
            labs_vid = torch.cat((labs_vid, labs_vid_temp), dim = 0)

            print("total data size:", data_vid.shape)
            print("total labs size:", labs_vid.shape)

            del data_vid_temp
            del labs_vid_temp


        cnt = cnt + 1

    print("")
    print("")
    print("=========================================================")
    print("loading trainning data completed!")
    print("trainning data info:")
    print("data size:", data_vid.shape)
    print("labs size:", labs_vid.shape)
    print("")
    print("")



    batchsize = 4000
    batchsize_detect = 4000

    num_epoch = qparams['epochnum']
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

    optim_ADVar = optim.Adam([W._f, B._f], lr = lr_rate, amsgrad = True)


    netC = ClassifyNetwork(num_dis).to(device)
    netP = PreproNetwork().to(device)

    optim_netC = optim.Adam(netC.parameters(), lr = lr_rate, amsgrad = True)
    optim_netP = optim.Adam(netP.parameters(), lr = lr_rate, amsgrad = True)

    class_weights = torch.FloatTensor([0.5, 0, 0.5]).to(device)
    loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='sum').to(device)




    print("")
    print("")
    print("=========================================================")
    print("process labs ...") 
    labs_vid = processLabs(labs_vid)
    print("processing completed:", torch.unique(labs_vid))
    
    print(torch.sum(labs_vid == 2))
    print(torch.sum(labs_vid == 1))
    print(torch.sum(labs_vid == 0))


    print("=========================================================")
    print("start training...")
    


    if preload > -1:

        epoch = preload
        
        name_netC = net_pa + 'netC_' + str(epoch).zfill(4) + '.pt'
        name_netP = net_pa + 'netP_' + str(epoch).zfill(4) + '.pt'


        name_f_W = net_pa + 'f_W_' + str(epoch).zfill(4) + '.pt'
        name_f_B = net_pa + 'f_B_' + str(epoch).zfill(4) + '.pt'


        W._f = torch.load(name_f_W, map_location=device)
        B._f = torch.load(name_f_B, map_location=device)

        netC.load_state_dict(torch.load(name_netC, map_location=device))
        netP.load_state_dict(torch.load(name_netP, map_location=device))


    for epoch in range(num_epoch):

        setupSeed(epoch)

        idx = torch.randperm(data_vid.shape[0])
        total_loss = 0

        # python自带越界保护，只入不舍。但是python 0.5 实际比0.5 要大，出现0时会加1,因此为0.499999999999999999
        for i in range(round(data_vid.shape[0]/batchsize + 0.499999999999999999)):
            data = data_vid[ idx[i*batchsize:(i + 1)*batchsize]].to(device, dtype = torch.float32)
            labs = labs_vid[ idx[i*batchsize:(i + 1)*batchsize]].to(device, dtype = torch.int64)

            X._f = netP(data)

            output_W = arithmeticDis(X, W, F_W, proddis)
            output_B = arithmeticDis(X, B, F_B, sumdis)

            output_DIS = (output_W._f + output_B._f)
            output_labs = netC(output_DIS)

            loss = loss_func(output_labs, labs)

 
            print("epoch:", epoch,  "  batch num:", i, "\\" , round(data_vid.shape[0]/batchsize + 0.499999999999999999),   " loss = ", loss.item(), end = '\r')
            rate = data.shape[0]/data_vid.shape[0]
            total_loss = total_loss + loss.item()*rate


            optim_ADVar.zero_grad()
            optim_netC.zero_grad()
            optim_netP.zero_grad()

            loss.backward(retain_graph = True)

            optim_ADVar.step()
            optim_netC.step()
            optim_netP.step()


        print("")
        print("epoch:", epoch, 'total_loss = ', total_loss)

        
        if epoch % 1 == 0:
            name_netC = net_pa + 'netC_' + str(epoch).zfill(4) + '.pt'
            name_netP = net_pa + 'netP_' + str(epoch).zfill(4) + '.pt'


            name_f_W = net_pa + 'f_W_' + str(epoch).zfill(4) + '.pt'
            name_f_B = net_pa + 'f_B_' + str(epoch).zfill(4) + '.pt'

            
            saveMod(name_f_W, W._f)
            saveMod(name_f_B, B._f)
            torch.save(netC.state_dict(), name_netC)
            torch.save(netP.state_dict(), name_netP)


        


    print("")
    print("")
    print("=========================================================")
    print("validate training data...")
    print("data_vid.shape:", data_vid.shape)
    print("labs_vid.shape:", labs_vid.shape)
    
    testlabs = detectFg(data_vid, batchsize_detect, device, X, W, B, F_W, F_B, proddis, sumdis, netP, netC)



    labs_vid = labs_vid/2
    testlabs = testlabs/2
   
    
    labs_vid = torch.round(labs_vid*255.0)
    testlabs = torch.round(testlabs*255.0)


    TP, FP, TN, FN = evaluation_numpy_entry_torch(labs_vid, testlabs)


    Re = TP/max((TP + FN), 1)
    Pr = TP/max((TP + FP), 1)

    Fm = (2*Pr*Re)/max((Pr + Re), 0.0001)

    print("TP:", TP)
    print("FP:", FP)
    print("TN:", TN)
    print("FN:", FN)

    print("Re:", Re)
    print("Pr:", Pr)
    print("Fm:", Fm)

    print("completed")





if __name__ == '__main__':

    argc = len(sys.argv)
    argv = sys.argv

    main(argc, argv)
