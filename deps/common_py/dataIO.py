"""Small file and tensor I/O helpers used by the LTS experiments.

The original project exposes these functions with camel-case names, and the
notebook imports them directly.  The public names are intentionally kept for
backwards compatibility; the implementation is only organised into a few
small, reusable helpers.
"""

import os

import torch


def _imageio():
    """Import imageio only when an image operation is requested."""

    import imageio.v2 as imageio

    return imageio


def loadFiles_plus(path_im, keyword=""):
    """Return sorted matching file names and their paths.

    Matching intentionally uses substring search, preserving the behaviour of
    the original helper (for example, ``keyword="png"`` also matches a file
    whose name contains ``png`` elsewhere).
    """

    re_fs = []
    re_fullfs = []

    for file in sorted(os.listdir(path_im)):
        if file.find(keyword) != -1:
            re_fs.append(file)
            # Keep the historical path format used by the experiment scripts.
            re_fullfs.append(path_im + "/" + file)

    return re_fs, re_fullfs


def saveImg(filename, im):
    """Save an image, creating its parent directory when needed."""

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    _imageio().imwrite(filename, im)


def saveMod(filename, mod):
    """Serialize a PyTorch object, creating its parent directory."""

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(mod, filename)


def checkCreateDir(filename):
    """Create the parent directory for ``filename`` if it does not exist."""

    os.makedirs(os.path.dirname(filename), exist_ok=True)


def readImg_byFilesIdx(curidx, path_im, keyword=""):
    """Read the ``curidx``-th sorted image from a directory."""

    _fs, fullfs = loadFiles_plus(path_im, keyword)
    return _imageio().imread(fullfs[curidx])


def loadImgs_pytorch_src(path_im, keyword=""):
    """Load all matching images into a tensor using the original dtype."""

    _fs, fullfs = loadFiles_plus(path_im, keyword)
    im = torch.tensor(_imageio().imread(fullfs[0]))

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
            im = torch.tensor(_imageio().imread(fullfs[i]))
            imgs_data[i] = im
        except Exception:
            print("files error:", fullfs[i])

    return imgs_data


def readImg_byFilesIdx_pytorch(curidx, path_im, keyword=""):
    """Read one image and convert it to a floating-point tensor."""

    _fs, fullfs = loadFiles_plus(path_im, keyword)
    return torch.tensor(_imageio().imread(fullfs[curidx]), dtype=torch.float)


def getVideoSize(pa_im, ft_im):
    """Return ``(frames, rows, columns, channels)`` for a video directory."""

    _fs, fullfs = loadFiles_plus(pa_im, ft_im)
    im = torch.tensor(_imageio().imread(fullfs[0]), dtype=torch.float)

    frames = len(fullfs)
    row_im, column_im, byte_im = im.shape

    return frames, row_im, column_im, byte_im


def loadImgs_pytorch(path_im, keyword=""):
    """Load all matching images into a floating-point PyTorch tensor."""

    _fs, fullfs = loadFiles_plus(path_im, keyword)
    im = torch.tensor(_imageio().imread(fullfs[0]), dtype=torch.float)

    row = column = byte = 1
    frame = len(fullfs)

    if len(im.shape) == 2:
        row, column = im.shape

    if len(im.shape) == 3:
        row, column, byte = im.shape

    imgs_data = torch.empty([frame, row, column, byte], dtype=torch.float).squeeze()

    for i in range(frame):
        print("load files:", fullfs[i], end="\r")
        im = torch.tensor(_imageio().imread(fullfs[i]), dtype=torch.float)
        imgs_data[i] = im

    print("")
    print("complete")

    return imgs_data


def showHello():
    """Print the historical smoke-test message."""

    print("in function hello")


def loadSubimgs_pytorch(path_im, keyword="", nums=1024):
    """Load at most ``nums`` matching images as floating-point tensors."""

    fs, fullfs = loadFiles_plus(path_im, keyword)
    fs = fs[:nums]
    fullfs = fullfs[:nums]

    im = torch.tensor(_imageio().imread(fullfs[0]), dtype=torch.float)

    row = column = byte = 1
    frame = len(fullfs)

    if len(im.shape) == 2:
        row, column = im.shape

    if len(im.shape) == 3:
        row, column, byte = im.shape

    imgs_data = torch.empty([frame, row, column, byte], dtype=torch.float).squeeze()

    for i in range(frame):
        print("load files:", fullfs[i], end="\r")
        im = torch.tensor(_imageio().imread(fullfs[i]), dtype=torch.float)
        imgs_data[i] = im

    print("")
    print("complete")

    return imgs_data
