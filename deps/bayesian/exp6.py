import sys


import time

# sys.path.append("/home/cqzhao/projects/matrix/")
sys.path.append("D:/projects/matrix")
import torch
import os

import imageio

from common_py.dataIO import readImg_byFilesIdx
from common_py.dataIO import loadFiles_plus

from skimage import color

import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

from common_py.evaluationBS import evaluation_numpy
from common_py.evaluationBS import evaluation_numpy_entry


# def img2pos(im):
#     row_im, column_im, byte_im = im.shape
#
#     re_pos = torch.abs(im - im)
#     re_pos = re_pos[:, :, 0:2]
#
#     for r in range(row_im):
#         for c in range(column_im):
#             re_pos[r, c, 0] = r
#             re_pos[r, c, 1] = c
#
#
#     return re_pos


def img2pos(im):
    row_im, column_im, byte_im = im.shape

    re_pos = torch.abs(im - im)
    re_pos = re_pos[:, :, 0:2]


    pos_r = torch.linspace(0, row_im - 1, row_im ).expand(column_im, row_im).t()
    pos_c = torch.linspace(0, column_im - 1, column_im).expand(row_im, column_im)

    return torch.cat((pos_r.unsqueeze(-1), pos_c.unsqueeze(-1)), dim = 2)





def rgb2lab(im):

    im = im/255.0
    lab = color.rgb2lab(im)

    lab = torch.tensor( lab, dtype = torch.float )



    return lab


def isnan(x):
    return x != x



def bayesRefine(im, fgim, radius, rate):

    row_im, column_im, byte_im = im.shape

    lab = rgb2lab(im)
    impos = img2pos(im)


    row, column, byte = im.shape

    features = torch.cat((im, im[:, :, 0:2]), dim = 2)
    features[:, :, 0] = lab[:, :, 0]
    features[:, :, 1] = lab[:, :, 1]
    features[:, :, 2] = lab[:, :, 2]
    features[:, :, 3] = impos[:, :, 0]
    features[:, :, 4] = impos[:, :, 1]


    [row_f, column_f, byte_f] = features.shape



    ex_feats = F.pad(features.permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode='replicate').squeeze()
    ex_feats = ex_feats.permute(1, 2, 0)

    ex_label = F.pad(fgim.unsqueeze(-1).permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode='replicate').squeeze()
    ex_label = ex_label


    ex_image = F.pad(im.permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode='replicate').squeeze()
    ex_image = ex_image.permute(1, 2, 0)


#     print(ex_feats.shape)
#     print(ex_label.shape)
#     print(ex_image.shape)

    num_feats = radius*2 + 1
    num_feats = num_feats*num_feats

#    print("num_feats = ", num_feats)

    store_label = ex_label

    cnt = 0
    for i in range(radius, row_f + radius):

        print("i = ", i)

        for j in range(radius, column_f + radius):

#            print("i,j:",i, j)

            entim = ex_feats[i - radius:i + radius + 1, j - radius:j + radius + 1, :]
            pixel = ex_feats[i, j, :]
            pixel = pixel.squeeze()

            label = ex_label[i - radius:i + radius + 1, j - radius:j + radius + 1]

            vec_im = entim.reshape(num_feats, byte_f)
            vec_lb = label.reshape(num_feats, 1)

            midpos = round(num_feats/2)
            vec_im = torch.cat((vec_im[0:midpos], vec_im[midpos + 1:num_feats]), dim = 0)
            vec_lb = torch.cat((vec_lb[0:midpos], vec_lb[midpos + 1:num_feats]), dim = 0)

            idx0 = vec_lb == 0
            idx1 = vec_lb == 255
            idx0 = idx0.squeeze()
            idx1 = idx1.squeeze()

            P0 = torch.sum(idx0).float()/(torch.sum(idx0) + torch.sum(idx1)).float()
            P1 = torch.sum(idx1).float()/(torch.sum(idx0) + torch.sum(idx1)).float()


#             if (P0 == 0) and (P1 == 0):
#                 print("P0:", P0)
#                 print("P1:", P1)
#
#                 print(vec_lb)
#
#                 print(idx0)
#                 print(idx1)
#
#
#                 print(torch.sum(idx0)/(torch.sum(idx0) + torch.sum(idx1)))
#                 print(torch.sum(idx1)/(torch.sum(idx0) + torch.sum(idx1)))
#
#                 print(torch.sum(idx0)*1.0/(torch.sum(idx0) + torch.sum(idx1))*1.0 )
#                 print(torch.sum(idx1)*1.0/(torch.sum(idx0) + torch.sum(idx1))*1.0 )
#
#                 print(torch.sum(idx0).float()/(torch.sum(idx0) + torch.sum(idx1)).float() )
#                 print(torch.sum(idx1).float()/(torch.sum(idx0) + torch.sum(idx1)).float() )
#




#                 t = input()

#             print("P0:", P0)
#             print("P1:", P1)

            if (P0 == 1) or (P1 == 1):
                store_label[i, j] = (P1 > rate*P0)*255
            else:
                distance = vec_im - pixel

                clrlist = distance[:, 0:3].mul(distance[:, 0:3])
                clrlist = torch.sum(clrlist, dim = 1)
                clrlist = torch.sqrt(clrlist)
                mean_clr_dis = torch.mean(clrlist)


                spalist = distance[:, 3:5].mul(distance[:, 3:5])
                spalist = torch.sum(spalist, dim = 1)
                spalist = torch.sqrt(spalist)
                mean_spa_dis = torch.mean(spalist)


#                print(idx0.shape)
#                print(vec_im.shape)
                vec_neg = vec_im[idx0, :]
                dis_neg = distance[idx0, :]
                mean_dis_neg = torch.mean(dis_neg, dim = 0)

                clr_dis_neg = torch.sqrt(torch.sum(mean_dis_neg[0:3].mul(mean_dis_neg[0:3])))
                spa_dis_neg = torch.sqrt(torch.sum(mean_dis_neg[3:5].mul(mean_dis_neg[3:5])))



                vec_pos = vec_im[idx1, :]
                dis_pos = distance[idx1, :]
                mean_dis_pos = torch.mean(dis_pos, dim = 0)

                clr_dis_pos = torch.sqrt(torch.sum(mean_dis_pos[0:3].mul(mean_dis_pos[0:3])))
                spa_dis_pos = torch.sqrt(torch.sum(mean_dis_pos[3:5].mul(mean_dis_pos[3:5])))


                mean_neg = torch.mean(vec_neg, dim = 0)
                mean_pos = torch.mean(vec_pos, dim = 0)

                sigma_neg = torch.mean( (vec_neg - torch.mean(vec_neg, dim = 0)).mul(vec_neg - torch.mean(vec_neg, dim = 0)), dim = 0 )
                sigma_pos = torch.mean( (vec_pos - torch.mean(vec_pos, dim = 0)).mul(vec_pos - torch.mean(vec_pos, dim = 0)), dim = 0 )

                sigma_dis_neg = torch.sqrt(sigma_neg[3] + sigma_pos[4])
                sigma_dis_pos = torch.sqrt(sigma_pos[3] + sigma_pos[4])


#                print(sigma_dis_neg)
#                print(sigma_dis_pos)


                if isnan(sigma_dis_neg):
                    sigma_dis_neg = 10000000

                if isnan(sigma_dis_pos):
                    sigma_dis_pos = 10000000


                nor_clr_dis_neg = clr_dis_neg/mean_clr_dis
                nor_spa_dis_neg = spa_dis_neg/mean_spa_dis

                nor_clr_dis_pos = clr_dis_pos/mean_clr_dis
                nor_spa_dis_pos = spa_dis_pos/mean_spa_dis

                nor_spa_dis_pos = nor_spa_dis_pos * (sigma_dis_pos/sigma_dis_neg)


                dis_neg = torch.sqrt(nor_clr_dis_neg*nor_clr_dis_neg + nor_spa_dis_neg*nor_spa_dis_neg)
                dis_pos = torch.sqrt(nor_clr_dis_pos*nor_clr_dis_pos + nor_spa_dis_pos*nor_spa_dis_pos)


                if isnan(dis_neg):
                    dis_neg = 10000000

                if isnan(dis_pos):
                    dis_pos = 10000000



                if (dis_neg == 0) or (dis_pos == 0) or (sigma_dis_pos == 0) or (sigma_dis_neg == 0):
                    dis_neg = 1
                    dis_pos = 1

#                print("dis_neg:", dis_neg)
#                print("dis_pos:", dis_pos)

                P0_X = P0 * (1/dis_neg)
                P1_X = P1 * (1/dis_pos)

#                print("")

#                print("borderline -----------------")
#                print("P1_X:", P1_X)
#                print("P0_X:", P0_X)
#                print("borderline -----------------")
                store_label[i,j] = (P1_X > rate*P0_X)*255

    re_fg = store_label[radius:radius + row_im, radius:radius + column_im]

    return re_fg

def bayesRefine_iterative_gpu(im, fgim, radius, rate, num, device):
    impos = img2pos(im)
    lab   = rgb2lab(im)

    im    = im.to(device)
    fgim  = fgim.to(device)
    impos = impos.to(device)
    lab   = lab.to(device)

    bayfgim = fgim
    for i in range(num):
        bayfgim = bayesRefine_torch(im, bayfgim, impos, lab, radius, rate)

    return bayfgim






def bayesRefine_torch(im, fgim, impos, lab, radius, rate):


    row_im, column_im, byte_im = im.shape


#    impos = img2pos(im)
#    lab = rgb2lab(im)



    row, column, byte = im.shape

    features = torch.cat((im, im[:, :, 0:2]), dim = 2)
    features[:, :, 0] = lab[:, :, 0]
    features[:, :, 1] = lab[:, :, 1]
    features[:, :, 2] = lab[:, :, 2]
    features[:, :, 3] = impos[:, :, 0]
    features[:, :, 4] = impos[:, :, 1]



    # normalization
#     features_vec = features.reshape(row_im*column_im, 5)
#
#     print("features_vec.shape:", features_vec.shape)
#
#
#     min_val, min_idx = torch.min(features_vec, dim = 0)
#     features_vec = features_vec - min_val
#
#
#     min_val, min_idx = torch.min(features_vec, dim = 0)
#
#     max_val, max_idx = torch.max(features_vec, dim = 0)
#
#     features_vec = features_vec/max_val
#
#     features = features_vec.reshape(row_im, column_im, 5)



    pixel_features = features


    [row_f, column_f, byte_f] = features.shape



    ex_feats = F.pad(features.permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode='replicate').squeeze()
    ex_feats = ex_feats.permute(1, 2, 0)

    ex_label = F.pad(fgim.unsqueeze(-1).permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode='replicate').squeeze()
    ex_label = ex_label


    ex_image = F.pad(im.permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode='replicate').squeeze()
    ex_image = ex_image.permute(1, 2, 0)


#     print(ex_feats.shape)
#     print(ex_label.shape)
#     print(ex_image.shape)

    num_feats = radius*2 + 1
    num_feats = num_feats*num_feats




    labs_vec = torch.cat([ex_label[i:i + row_im, j:j + column_im].reshape(row_im*column_im, 1) for i in range(radius*2 + 1) for j in range(radius*2 + 1)], dim = 1)
    imgs_vec = torch.cat([ex_feats[i:i + row_im, j:j + column_im].reshape(row_im*column_im, 5).unsqueeze(-1) for i in range(radius*2 + 1) for j in range(radius*2 + 1)], dim = 2)


    pixel_vec = pixel_features.reshape(row_im*column_im, 5)

    dis_vec = imgs_vec.permute(2, 0, 1) - pixel_vec


    midpos = round(num_feats/2)

    dis_vec  = torch.cat((dis_vec[0:midpos, :, :], dis_vec[midpos + 1:num_feats, :, :]) , dim = 0 )
    labs_vec = torch.cat((labs_vec[:, 0:midpos],   labs_vec[:, midpos + 1:num_feats]), dim = 1)
    imgs_vec = torch.cat((imgs_vec[:, :, 0:midpos], imgs_vec[:, :, midpos + 1:num_feats]), dim = 2)

    imgs_vec = imgs_vec.permute(0, 2, 1)


#     print("borderline =============")
#     print(dis_vec.shape)
#     print(labs_vec.shape)
#     print("borderline =============")
#     48 124416 5
#     124416 48



    P0_num = torch.sum(labs_vec == 0, dim = 1).float()
    P1_num = torch.sum(labs_vec == 255, dim = 1).float()


    P0 = P0_num/(P0_num + P1_num)
    P1 = P1_num/(P0_num + P1_num)


    imgs_vec_pos = imgs_vec.clone()
    imgs_vec_neg = imgs_vec.clone()

#    print(imgs_vec.shape)
#    print(labs_vec.shape)


    imgs_vec_pos[labs_vec == 0]   = 0
    imgs_vec_neg[labs_vec == 255] = 0




    mus_vec_pos = (imgs_vec_pos.sum(dim = 1).permute(1, 0)/P1_num).permute(1, 0)
    mus_vec_neg = (imgs_vec_neg.sum(dim = 1).permute(1, 0)/P0_num).permute(1, 0)





    sigma_vec_pos = (imgs_vec.permute(1, 0, 2) - mus_vec_pos).permute(1, 0, 2)
    sigma_vec_neg = (imgs_vec.permute(1, 0 ,2) - mus_vec_neg).permute(1, 0, 2)

    sigma_vec_pos[labs_vec == 0] = 0
    sigma_vec_pos = torch.abs(sigma_vec_pos)
#    sigma_vec_pos = sigma_vec_pos.mul(sigma_vec_pos).sum(dim = 1).permute(1, 0)/P1_num
    sigma_vec_pos = sigma_vec_pos.sum(dim = 1).permute(1, 0)/P1_num
    sigma_vec_pos = sigma_vec_pos.permute(1, 0)

    sigma_vec_neg[labs_vec == 255] = 0
    sigma_vec_neg = torch.abs(sigma_vec_neg)
#    sigma_vec_neg = sigma_vec_neg.mul(sigma_vec_neg).sum(dim = 1).permute(1, 0)/P0_num
    sigma_vec_neg = sigma_vec_neg.sum(dim = 1).permute(1, 0)/P0_num
    sigma_vec_neg = sigma_vec_neg.permute(1, 0)




    sigma_vec_pos[sigma_vec_pos == 0] = 100000
    sigma_vec_neg[sigma_vec_neg == 0] = 100000


    # for 0








    dis_vec_pos = mus_vec_pos - pixel_vec
    dis_vec_neg = mus_vec_neg - pixel_vec


    abs_dis_vec_pos = torch.abs(dis_vec_pos)
    abs_dis_vec_neg = torch.abs(dis_vec_neg)


    pro_vec_pos = 1 - abs_dis_vec_pos.mul(0.5/sigma_vec_pos)
    pro_vec_neg = 1 - abs_dis_vec_neg.mul(0.5/sigma_vec_neg)



    avg_pro_pos = pro_vec_pos[:,0]*0.2 + pro_vec_pos[:,1]*0.2 + pro_vec_pos[:,2]*0.2 + pro_vec_pos[:,3]*0.2 + pro_vec_pos[:,4]*0.2
    avg_pro_neg = pro_vec_neg[:,0]*0.2 + pro_vec_neg[:,1]*0.2 + pro_vec_neg[:,2]*0.2 + pro_vec_neg[:,3]*0.2 + pro_vec_neg[:,4]*0.2


    clr_pro_vec_pos = pro_vec_pos[:, 0:3]
    # spa_pro_vec_pos = pro_vec_pos[:, 3:5].max(dim = 1)[0]
    spa_pro_vec_pos = pro_vec_pos[:, 3:5]


    clr_pro_vec_neg = pro_vec_neg[:, 0:3]
    #spa_pro_vec_neg = pro_vec_neg[:, 3:5].max(dim = 1)[0]
    spa_pro_vec_neg = pro_vec_neg[:, 3:5]



    avg_pro_pos = clr_pro_vec_pos[:, 0]*0.3 + clr_pro_vec_pos[:, 1]*0.105 + clr_pro_vec_pos[:, 2]*0.105 + spa_pro_vec_pos[:, 0]*0.245 + spa_pro_vec_pos[:, 1]*0.245
    avg_pro_neg = clr_pro_vec_neg[:, 0]*0.3 + clr_pro_vec_neg[:, 1]*0.105 + clr_pro_vec_neg[:, 2]*0.105 + spa_pro_vec_neg[:, 0]*0.245 + spa_pro_vec_neg[:, 1]*0.245


    P0_X = P0.mul(avg_pro_neg)
    P1_X = P1.mul(avg_pro_pos)


#    P1_X = P1 + avg_pro_pos
#    P0_X = P0 + avg_pro_neg


#    avg_pro_pos = clr_pro_vec_pos[:, 0]*0.5 + spa_pro_vec_pos*0.5
#    avg_pro_neg = clr_pro_vec_neg[:, 0]*0.5 + spa_pro_vec_neg*0.5


#    t = pro_dis_vec_pos.max(dim = 1)



#     print("borderline ===============================================================")
#
#     idx1 = 45244
#
#     print(abs_dis_vec_pos[idx1])
#     print(abs_dis_vec_neg[idx1])
#
#     print(sigma_vec_pos[idx1])
#     print(sigma_vec_neg[idx1])
#
#
# #     print(clr_pro_vec_pos.shape)
# #     print(spa_pro_vec_pos.shape)
# #
# #     print(clr_pro_vec_neg.shape)
# #     print(spa_pro_vec_neg.shape)
#
#
#
#     print(pro_vec_pos[idx1])
#     print(pro_vec_neg[idx1])
#
#
#
#     print(spa_pro_vec_pos[idx1])
#     print(spa_pro_vec_neg[idx1])
#
#
#     print(avg_pro_pos[idx1])
#     print(avg_pro_neg[idx1])
#
#     print(P1[idx1])
#     print(P0[idx1])
#
# #     print(dis_vec_pos[idx1])
# #     print(dis_vec_neg[idx1])
# #     print(sigma_vec_pos[idx1])
# #     print(sigma_vec_neg[idx1])
# #
# #     print(P1[idx1])
# #     print(P0[idx1])
# #
# #     print("")
# #     print("--------------------------------------")
# #     print(avg_pro_pos[idx1])
# #     print(avg_pro_neg[idx1])
# #     print(P0[idx1])
# #     print(P1[idx1])
#
#     print("borderline ===============================================================")
#



    re_labs = torch.abs(fgim - fgim).reshape(row_im*column_im)
    re_labs = (P1_X > rate*P0_X)*255

    re_labs[P0 > 0.8] = 0
    re_labs[P1 > 0.8] = 255


    bayfgim = re_labs.reshape(row_im, column_im).float()


    return bayfgim
















def main():


    im_pa = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/input'
    im_ft = 'jpg'

#    fg_pa = 'D:/projects/lab/fgimgs_fountain01_v34'
    fg_pa = 'D:/projects/lab/fgimgs_fountain01_v49_0'
    fg_ft = 'png'

    gt_pa = 'D:/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth'
    gt_ft = 'png'






    use_cuda = torch.cuda.is_available()


    print("------------")
    print(use_cuda)
    print("------------")

    torch.manual_seed(0)

    device = torch.device("cuda:0" if use_cuda else "cpu")




    fs_im, fullfs_im = loadFiles_plus(im_pa, im_ft)
    fs_fg, fullfs_fg = loadFiles_plus(fg_pa, fg_ft)
    fs_gt, fullfs_gt = loadFiles_plus(gt_pa, gt_ft)


    idx = 1148
    idx = 753
    idx = 1104

    im =   torch.tensor( imageio.imread(fullfs_im[idx]), dtype = torch.float )
    fgim = torch.tensor( imageio.imread(fullfs_fg[idx]), dtype = torch.float )
    gtim = torch.tensor( imageio.imread(fullfs_gt[idx]), dtype = torch.float )




    # start here

    radius = 1
    rate = 0.65
    num = 20

#     im = im.to(device)
#     fgim = fgim.to(device)
# def bayesRefine_torch(im, fgim, impos, lab, radius, rate, device):


#    row_im, column_im, byte_im = im.shape

    starttime = time.time()
    bayfgim = bayesRefine_iterative_gpu(im, fgim, radius, rate, num, device)
    endtime = time.time()

    print("total time:", endtime - starttime)


#    bayfgim = bayesRefine_torch(im, bayfgim, radius, rate)
#    bayfgim = bayesRefine_torch(im, bayfgim, radius, rate)
#    bayfgim = bayesRefine_torch(im, bayfgim, radius, rate)
#    bayfgim = bayesRefine_torch(im, bayfgim, radius, rate)





    fgim = fgim.detach().cpu().numpy()
    gtim = gtim.detach().cpu().numpy()
    bayfgim = bayfgim.detach().cpu().numpy()

    im = im.detach().cpu().numpy()


    Re, Pr, Fm = evaluation_numpy(np.round(fgim),    np.round(gtim))

    print("Re, Pr, Fm:", Re, Pr, Fm)

    Re, Pr, Fm = evaluation_numpy(np.round(bayfgim), np.round(gtim))

    print("Re, Pr, Fm:", Re, Pr, Fm)



    plt.figure()
    plt.subplot(1, 3, 1)
#    plt.imshow(fgim.detach().cpu().numpy(), cmap='gray')
    plt.imshow(fgim, cmap='gray')

    plt.subplot(1, 3, 2)
#    plt.imshow(bayfgim.detach().cpu().numpy(), cmap='gray')

    plt.imshow(bayfgim, cmap='gray')

    plt.subplot(1, 3, 3)

    plt.imshow(im/255.0)


    plt.show()








#
#     print(clr_dis_pos_sum[idx])
#     print(clr_dis_neg_sum[idx])
#     print(spa_dis_pos_sum[idx])
#     print(spa_dis_neg_sum[idx])
#
#
#     print(clr_dis_pos[idx, :])
#     print(clr_dis_neg[idx, :])
#
#     print(spa_dis_pos[idx, :])
#     print(spa_dis_neg[idx, :])
#
#
#     print(P0_num[idx])
#     print(P1_num[idx])

#     print(labs_vec.shape)
#     print(mean_clr_dis.shape)
#     print(mean_spa_dis.shape)
#
#     print(clr_dis_vec[:, idx])
#     print(spa_dis_vec[:, idx])
#
#     print(P0[idx])
#     print(P1[idx])
#
#
#     print(clr_dis_vec.shape)
#     print(spa_dis_vec.shape)
#
#








#     neg_dis = dis_vec.clone()
#     pos_dis = dis_vec.clone()
#
#     neg_dis[idxlabs == 255] = 0
#     pos_dis[idxlabs == 0] = 0
#
#
#
#     idx = 42032
#
#     print(neg_dis[:, idx, :])
#     print(pos_dis[:, idx, :])
#
#
#
#     print(neg_dis.shape)
#     print(pos_dis.shape)


#     neg_dis = neg_dis.sum(dim = 0).sum(dim = 1)
#     pos_dis = pos_dis.sum(dim = 0).sum(dim = 1)

#
#     idx = 42032
#
#     print(neg_dis[:, idx, 0])
#     print(pos_dis[:, idx, 0])
#
#     print(idxlabs[:, idx, 0])


#     neg_dis = neg_dis/P0_num
#     pos_dis = pos_dis/P1_num
#
#
#
#     neg_dis = 1.0/neg_dis
#     pos_dis = 1.0/pos_dis
#
#
#     neg_pra = neg_dis*P0
#     pos_pra = pos_dis*P1
#
#
#     re_labs = torch.abs(fgim - fgim).reshape(row_im*column_im)
#     re_labs[pos_pra > neg_pra] = 255
#     re_labs[P0_num == 0] = 255
#     re_labs[P1_num == 0] = 0
#
#     bayfgim = re_labs.reshape(row_im, column_im)
#
#
#     print(bayfgim.shape)
#
#
#     print(P0_num.shape)
#     print(P1_num.shape)
#
#
#     print(neg_dis.shape)
#     print(pos_dis.shape)
#
#
#
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(fgim.detach().cpu().numpy(), cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.imshow(bayfgim.detach().cpu().numpy(), cmap='gray')
#
#     plt.show()
#
#
#
#












#
#     sum_dis = torch.abs(dis_vec).sum(dim = 0).sum(dim = 1)
#
#
#
#     idx = 42032
#
#     print(neg_dis[:, idx, 0])
#     print(pos_dis[:, idx, 0])
#
#     print(idxlabs[:, idx, 0])






#
#
#
#
#     print(neg_dis.shape)
#     print(pos_dis.shape)
#
#
#
#     print(dis_vec.shape)
#     print(labs_vec.shape)
#     print(idxlabs.shape)



#     print(dis_vec.shape)
#     print(labs_vec.shape)
#
#
#     dis_vec = dis_vec.permute(2, 1, 0)
#     idx = labs_vec == 0
#
#     print("dis_vec.shape:", dis_vec.shape)
#
#     dis_vec[:, idx] = 0
#
#     print(dis_vec.shape)


#    dis_neg = torch.sum(dis_vec[:, idx], dim = 2)

#    print(dis_neg.shape)






#
#     for i in range(124416):
#         print(P0[i])
#         print(torch.sum(labs_vec[i,:] == 0))
#
#         print("")



#    print(labs_vec.shape)
#    print(imgs_vec.shape)
#    print(pixel_vec.shape)




#     for i in range(radius*2 + 1):
#         for j in range(radius*2 + 1):
#             labs = ex_label[i:i + row_im, j:j + column_im].reshape(row_im*column_im, 1)
#             print(labs.shape)
# #            print(t.shape)
#
#
# #    print(labs)
#     print(torch.sum( lab_pad[:, 48] - labs.squeeze()))



if __name__ == '__main__':
    main()
