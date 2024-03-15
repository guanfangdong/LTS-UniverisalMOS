import torch

from skimage import color
import torch.nn.functional as F



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



# 以每个像素为中心，取出一个radius为10的方块patch
def imPatching(im, radius = 10):

    row_im, column_im, byte_im = im.shape
#    radius = 10

    exim = F.pad(im.permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode='replicate').squeeze()
    exim = exim.permute(1, 2, 0)

    imgs_vec = torch.cat([exim[i:i + row_im, j:j + column_im].reshape(row_im*column_im, byte_im).unsqueeze(-1) for i in range(radius*2 + 1) for j in range(radius*2 + 1)], dim = 2)

    nums, byte, cha = imgs_vec.shape
    imgs_vec = imgs_vec.reshape(row_im, column_im, byte, 2*radius + 1, 2*radius + 1)

    return imgs_vec.permute(0, 1, 3, 4, 2)



def im2patch(im, size = (64, 64)):

    row_im, column_im, byte_im = im.shape


    if row_im%size[0] == 0:
        row_up = row_im - row_im%size[0]
    else:
        row_up = row_im - row_im%size[0] + size[0]

    if column_im%size[1] == 0:
        column_up = column_im - column_im%size[1]
    else:
        column_up = column_im - column_im%size[1] + size[1]


    pad_r = (row_up - row_im)//2
    pad_c = (column_up - column_im)//2
  

    exim = F.pad(im.permute(2, 0, 1).unsqueeze(0), (pad_c, pad_c, pad_r, pad_r), mode='constant').squeeze()
    exim = exim.permute(1, 2, 0)

    
    num_r = exim.shape[0]//size[0]
    num_c = exim.shape[1]//size[1]


    patches = torch.empty(num_r*num_c, size[0], size[1], byte_im)

    cnt = 0
    for i in range(num_r):
        for j in range(num_c):
            
            patches[cnt] = exim[i*size[0]:(i + 1)*size[0], j*size[1]:(j + 1)*size[1], :]
            cnt = cnt + 1


    return patches


def checkInPad(t, b, l ,r, row_im, column_im, row_rad, column_rad):

    if b > row_im:

        b = row_im
        t = b - row_rad


    if r > column_im:

        r = column_im
        l = r - column_rad


    return t, b, l, r



def im2patch_inpad(im, size = (64, 64)):

    row_im, column_im, byte_im = im.shape

    num_I = round(im.shape[0]/size[0] + 0.499999999999)
    num_J = round(im.shape[1]/size[1] + 0.499999999999)


    patches = torch.empty(num_I*num_J, size[0], size[1], byte_im)

    cnt = 0
    for i in range(num_I):
        for j in range(num_J):

            t = i*size[0]
            b = t + size[0]

            l = j*size[1]
            r = l + size[1]

            t, b, l ,r = checkInPad(t, b, l ,r, row_im, column_im, size[0], size[1])


            patches[cnt] = im[t:b, l:(j + 1)*size[1], :]
            cnt = cnt + 1

    return patches


def patch2im_inpad(patches, imgsize = (218, 178, 3)):

    row_im      = imgsize[0]
    column_im   = imgsize[1]
    byte_im     = imgsize[2]

    im = torch.empty(row_im, column_im, byte_im)

    size = (patches.shape[1], patches.shape[2])


    num_I = round(im.shape[0]/size[0] + 0.499999999999)
    num_J = round(im.shape[1]/size[1] + 0.499999999999)

    cnt = 0
    for i in range(num_I):
        for j in range(num_J):

            t = i*size[0]
            b = t + size[0]

            l = j*size[1]
            r = l + size[1]

            t, b, l ,r = checkInPad(t, b, l ,r, row_im, column_im, size[0], size[1])


            im[t:b, l:(j + 1)*size[1], :] = patches[cnt]
            cnt = cnt + 1


    return im



def im2patch_unique(im, size = (64, 64)):

    row_im, column_im, byte_im = im.shape

    num_I = round(im.shape[0]/size[0] + 0.499999999999)
    num_J = round(im.shape[1]/size[1] + 0.499999999999)


    patches = torch.empty(num_I*num_J, size[0], size[1], byte_im)


    
    cnt = 0
    for i in range(num_I):
        for j in range(num_J):

            t = i*size[0]
            b = t + size[0]

            l = j*size[1]
            r = l + size[1]

            t_c, b_c, l_c ,r_c = checkInPad(t, b, l ,r, row_im, column_im, size[0], size[1])

            if (t_c == t) and (b_c == b) and (l_c == l) and (r_c == r):

                patches[cnt] = im[t:b, l:(j + 1)*size[1], :]
                cnt = cnt + 1

    patches = patches[:cnt]

    return patches



def batchConv3d_trans(imgs, weights, stride = 2, padding = 0):

    num, in_chls, out_chls, row, column, byte = weights.shape
    
    imgs = imgs.reshape(1, -1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
    weights = weights.reshape(-1, out_chls, row, column, byte) 

    outimgs = F.conv_transpose3d(imgs, weights, stride = stride, padding = padding, groups = num)

    outimgs = outimgs.reshape(num, out_chls, outimgs.shape[2], outimgs.shape[3], outimgs.shape[4])


    return outimgs




def batchConv3d(imgs, weights, stride = 1, padding = 1):

    num, in_chls, out_chls, row, column, byte = weights.shape

    imgs = imgs.reshape(1, -1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
    weights = weights.reshape(-1, out_chls, row, column, byte)

    outimgs = F.conv3d(imgs, weights, stride = stride, padding = padding, groups = num)

    outimgs = outimgs.reshape(num, in_chls, outimgs.shape[2], outimgs.shape[3], outimgs.shape[4])


    return outimgs




def batchConv2d_trans(imgs, weights, stride = 2, padding = 0):

    num, in_chls, out_chls, row, column = weights.shape

    imgs = imgs.reshape(1, -1, imgs.shape[2], imgs.shape[3])
    weights = weights.reshape(-1, out_chls, row, column)

    outimgs = F.conv_transpose2d(imgs, weights, stride = stride, padding = padding, groups = num)

    outimgs = outimgs.reshape(num, out_chls, outimgs.shape[2], outimgs.shape[3])

 
    return outimgs


def batchConv2d(imgs, weights, stride = 1, padding = 1):

    num, in_chls, out_chls, row, column = weights.shape

    imgs = imgs.reshape(1, -1, imgs.shape[2], imgs.shape[3])
    weights = weights.reshape(-1, out_chls, row, column)

    outimgs = F.conv2d(imgs, weights, stride = stride, padding = padding, groups = num)

    outimgs = outimgs.reshape(num, in_chls, outimgs.shape[2], outimgs.shape[3])


    return outimgs


def quantize(data, delta=0.01, exp=2):
    
    return torch.round( data/(delta) + delta**exp )*delta
