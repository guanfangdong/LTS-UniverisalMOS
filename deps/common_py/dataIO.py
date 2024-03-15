import os
import torch

import imageio


def loadFiles_plus(path_im, keyword = ""):
    re_fs = []
    re_fullfs = []

    files = os.listdir(path_im)
    files = sorted(files)

    for file in files:
        if file.find(keyword) != -1:
            re_fs.append(file)
            re_fullfs.append(path_im + "/" + file)

    return re_fs, re_fullfs


def saveImg(filename, im):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    imageio.imwrite(filename, im)


def saveMod(filename, mod):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(mod, filename)



def checkCreateDir(filename):

    os.makedirs(os.path.dirname(filename), exist_ok=True)






def readImg_byFilesIdx(curidx, path_im, keyword = ""):

    fs, fullfs = loadFiles_plus(path_im, keyword)

    return imageio.imread(fullfs[curidx])






def loadImgs_pytorch_src(path_im, keyword = ""):

    fs, fullfs = loadFiles_plus(path_im, keyword)
    im = torch.tensor( imageio.imread(fullfs[0]))


    row = column = byte = 1
    frame = len(fullfs)


    if len(im.shape) == 2:
        row, column = im.shape

    if len(im.shape) == 3:
        row, column, byte = im.shape



    imgs_data = torch.empty([frame, row, column, byte], dtype=im.dtype).squeeze()

    for i in range(frame):
        print("load files:", fullfs[i])

        try:
            im = torch.tensor(imageio.imread(fullfs[i]))

            imgs_data[i] = im
        except:
            print("files error:", fullfs[i])


    return imgs_data


def readImg_byFilesIdx_pytorch(curidx, path_im, keyword = ""):

    fs, fullfs = loadFiles_plus(path_im, keyword)

    return torch.tensor( imageio.imread(fullfs[curidx]), dtype = torch.float )


def getVideoSize(pa_im, ft_im):
    fs, fullfs = loadFiles_plus(pa_im, ft_im)

    im = torch.tensor( imageio.imread(fullfs[0]), dtype = torch.float )

    frames = len(fullfs)

    row_im, column_im, byte_im = im.shape

    return frames, row_im, column_im, byte_im




def loadImgs_pytorch(path_im, keyword = ""):

    fs, fullfs = loadFiles_plus(path_im, keyword)
    im = torch.tensor( imageio.imread(fullfs[0]), dtype = torch.float)


    row = column = byte = 1
    frame = len(fullfs)


    if len(im.shape) == 2:
        row, column = im.shape

    if len(im.shape) == 3:
        row, column, byte = im.shape



    imgs_data = torch.empty([frame, row, column, byte], dtype=torch.float).squeeze()

    for i in range(frame):
        print("load files:", fullfs[i], end = '\r')
        im = torch.tensor(imageio.imread(fullfs[i]), dtype=torch.float)

        imgs_data[i] = im

    print("")
    print("complete")

    return imgs_data


def showHello():
    print("in function hello")




def loadSubimgs_pytorch(path_im, keyword = "", nums = 1024):

    fs, fullfs = loadFiles_plus(path_im, keyword)
    fs = fs[:nums]
    fullfs = fullfs[:nums]

    im = torch.tensor( imageio.imread(fullfs[0]), dtype = torch.float)


    row = column = byte = 1
    frame = len(fullfs)


    if len(im.shape) == 2:
        row, column = im.shape

    if len(im.shape) == 3:
        row, column, byte = im.shape



    imgs_data = torch.empty([frame, row, column, byte], dtype=torch.float).squeeze()

    for i in range(frame):
        print("load files:", fullfs[i], end = '\r')
        im = torch.tensor(imageio.imread(fullfs[i]), dtype=torch.float)

        imgs_data[i] = im

    print("")
    print("complete")

    return imgs_data

