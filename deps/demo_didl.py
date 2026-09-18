"""Reusable DIDL components used by the LTS demonstration notebook."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deps.common_py.dataIO import getVideoSize, readImg_byFilesIdx_pytorch
from deps.function.arithdis import arithmeticDis


class GaussianBlurConv(nn.Module):
    """
    This class implements a Gaussian Blur as a convolutional operation using a fixed kernel,
    designed to be applied on images. It extends the `nn.Module` class from PyTorch,
    allowing it to be integrated into a neural network as a layer.

    The Gaussian Blur is applied using a predefined kernel that approximates the Gaussian distribution.
    This kernel is applied to each channel of the input image separately, preserving the number of channels.

    Parameters:
    - channels (int): The number of channels in the input images. Defaults to 3 for RGB images.

    The convolution uses 'padding=2' to ensure the output image has the same height and width as the input image.
    """

    def __init__(self, channels=3):
        """
        Initializes the GaussianBlurConv layer.

        Parameters:
        - channels (int): The number of input channels (e.g., 3 for RGB images).
        """
        super().__init__()
        self.channels = channels

        # Define a 5x5 Gaussian kernel.
        kernel = [
            [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
            [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
            [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
            [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
            [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
        ]

        # Convert the kernel to a FloatTensor and reshape to fit the conv2d operation.
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, H, W)

        # Expand the kernel across the channel dimension to apply the same kernel to each channel.
        kernel = kernel.expand((int(channels), 1, 5, 5))

        # Register the kernel as a non-trainable parameter of the model.
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        """
        Applies the Gaussian Blur to the input image tensor.

        Parameters:
        - x (Tensor): The input image tensor of shape (batch_size, channels, height, width).

        Returns:
        - Tensor: The blurred image tensor of the same shape as the input.
        """
        # Apply the convolution operation with the Gaussian kernel.
        # Use padding to maintain the same image size and specify groups to apply the kernel to each channel separately.
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x


_DEFAULT_BLUR = GaussianBlurConv()


def batchConvImgs(imgs, conv=_DEFAULT_BLUR):
    """
    Applies a convolution operation to each image in a batch.

    This function iterates over a batch of images and applies a specified convolutional operation to each image. The operation is assumed to be an instance of a convolutional layer that expects input of the shape (batch_size, channels, height, width) and returns output of the same shape. Before applying the convolution, the function adjusts the shape of each image to match this expectation and then reshapes the output back to its original form.

    Parameters:
    - imgs (numpy.ndarray or torch.Tensor): A batch of images with shape (frames, row, column, byte), where 'frames' is the number of images, 'row' and 'column' are the dimensions of each image, and 'byte' is the number of channels per pixel.
    - conv (nn.Module): An instance of a PyTorch convolutional layer (or similar) to be applied to each image. Defaults to GaussianBlurConv(), a predefined Gaussian Blur convolution.

    Returns:
    - numpy.ndarray or torch.Tensor: The batch of images after applying the specified convolutional operation, with the same shape as the input.
    """

    # Extract the dimensions of the input image batch.
    frames_im, _row_im, _column_im, _byte_im = imgs.shape

    # Iterate over each image in the batch.
    for i in range(frames_im):
        # Reshape the image from (row, column, byte) to (1, byte, row, column) to match
        # the expected input shape for the convolutional layer. This involves adding a
        # batch dimension (unsqueeze) and permuting the dimensions.
        image = imgs[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 1, 2)

        # Apply the convolutional operation to the reshaped image.
        conv_image = conv(image)

        # Reshape the convolved image back to its original shape (row, column, byte)
        # by removing the batch dimension (squeeze) and permuting the dimensions back.
        imgs[i] = conv_image.squeeze().permute(1, 2, 0)

    # Return the batch of images with the convolutional operation applied.
    return imgs


def getVidHist_plus(imgs, left=-1, right=1, border=0.01):
    """
    Computes histograms for each pixel across all frames and channels of a batch of images.

    This function reshapes and permutes the input batch of images to compute histograms
    for each color channel of each pixel across all frames. It allows for specifying the
    range of values and the bin width (border) for the histograms. The histograms are normalized
    by the number of frames, providing a distribution of pixel values across the batch.

    Parameters:
    - imgs (torch.Tensor): A tensor representing a batch of images with shape (frame, row, column, byte),
      where 'frame' is the number of images, 'row' and 'column' are the image dimensions, and 'byte'
      is the number of channels (e.g., 3 for RGB).
    - left (float): The lower bound of the histogram range. Defaults to -1.
    - right (float): The upper bound of the histogram range. Defaults to 1.
    - border (float): The width of each histogram bin. Defaults to 0.01.

    Returns:
    - torch.Tensor: A tensor containing the histograms for each pixel and channel with shape
      (row*column, num_bins, byte), where 'num_bins' is the number of histogram bins determined
      by the range and bin width.
    """

    # Unpack the dimensions of the input batch of images.
    frame, row, column, byte = imgs.shape

    # Reshape and permute the images to have shape (row*column, frame, byte), which
    # facilitates computing histograms for each pixel location across all frames.
    imgs = imgs.reshape(frame, row * column, byte)
    imgs = imgs.permute(1, 0, 2)

    # Calculate the number of bins for the histograms based on the specified range and bin width.
    len_hist = round((right - left) / border) + 1

    # Initialize an empty tensor to store the histogram data for each pixel and channel.
    hist_data = torch.empty([row * column, len_hist, byte])

    # Compute histograms for each pixel location across all frames and for each channel.
    for i in range(row * column):
        for b in range(byte):
            # Normalize pixel values to the range [left, right] and compute the histogram
            # for the current pixel location and channel. The histogram counts are normalized
            # by the number of frames to obtain a distribution.
            hist_data[i, :, b] = torch.histc(
                (imgs[i, :, b] / 255.0) * right, len_hist, left, right
            ) / (frame * 1.0)

    # Return the computed histograms.
    return hist_data


def getNormalData_byHistVid_files(
    vid_hist, pa_im, ft_im, pa_gt, ft_gt, curidx, left=-1, right=1, delta=0.01
):
    """
    Generates normalized data and labels for a specific frame in a video, based on precomputed histograms of video data.

    This function processes a single frame by applying a convolution operation (assuming batchConvImgs is defined to do so),
    normalizing its pixel values, and mapping these values to their corresponding histogram bins. Additionally, it reads the
    ground truth image for the frame and prepares the label data. The function is useful for tasks that require normalized
    feature representations along with corresponding labels, such as supervised learning tasks in computer vision.

    Parameters:
    - vid_hist (torch.Tensor): A tensor containing the histograms for each pixel and channel across all frames of the video.
    - pa_im (str): The path to the directory containing the video frames.
    - ft_im (str): The file type or extension of the video frames.
    - pa_gt (str): The path to the directory containing the ground truth images.
    - ft_gt (str): The file type or extension of the ground truth images.
    - curidx (int): The index of the current frame to process.
    - left (float): The lower bound of the normalized value range. Defaults to -1.
    - right (float): The upper bound of the normalized value range. Defaults to 1.
    - delta (float): The bin width for normalization. Defaults to 0.01.

    Returns:
    - hist_data (torch.Tensor): A tensor of the same shape as vid_hist, containing normalized histogram data for the current frame.
    - labs_data (torch.Tensor): A tensor containing the labels for each pixel in the current frame, reshaped to a 1D array.
    """

    # Get the size of the video frames.
    _frames, row_im, column_im, byte_im = getVideoSize(pa_im, ft_im)

    # Read the current frame and apply the convolution operation.
    im = readImg_byFilesIdx_pytorch(curidx, pa_im, ft_im)
    im = batchConvImgs(im.unsqueeze(0)).squeeze()
    im = im.reshape(row_im * column_im, byte_im)

    # Read the ground truth image for the current frame.
    lb = readImg_byFilesIdx_pytorch(curidx, pa_gt, ft_gt)

    # Normalize the image pixel values and map them to histogram bins.
    im = (im / 255.0) * right
    im = torch.round(im / delta)

    # Compute the number of bins and offsets for histogram mapping.
    num_hist = round((right - left) / delta) + 1
    num_right = round(right / delta) + 1
    offset_right = round(right / delta)

    # Initialize histogram data tensor.
    hist_data = torch.abs(
        vid_hist - vid_hist
    )  # Creates a tensor of zeros with the same shape as vid_hist.
    labs_data = lb.reshape(row_im * column_im)  # Reshape label data to a 1D array.

    # Map normalized pixel values to their corresponding histogram bins.
    for i in range(num_right):
        for b in range(byte_im):
            idx_r = im[:, b] == i
            hist_data[idx_r, (num_hist - offset_right - i) : (num_hist - i), b] = vid_hist[
                idx_r, (num_hist - offset_right) : num_hist, b
            ]

    return hist_data, labs_data


class ClassifyNetwork(nn.Module):
    """
    A convolutional neural network (CNN) for classifying images into a fixed number of categories.

    The network consists of a single convolutional layer followed by two fully connected (linear) layers.
    The convolutional layer is designed to process 3-channel (RGB) images. The model applies a ReLU
    activation function after the first fully connected layer and uses a log softmax activation function
    on the output layer to provide log probabilities of each class.

    Attributes:
    - conv1 (nn.Conv2d): The first convolutional layer with 3 input channels, 1 output channel, and a kernel size of (1, 8).
    - fc1 (nn.Linear): The first fully connected layer that maps the flattened output of the convolutional layer to 512 features.
    - fc2 (nn.Linear): The second fully connected layer that maps the 512 features to the number of target classes (3 in this case).

    Parameters:
    - dis_num (int): The number of classes to classify. This parameter is currently unused in the initializer.
    """

    def __init__(self, dis_num):
        super().__init__()

        # Initialize the convolutional layer with specific parameters.
        self.conv1 = nn.Conv2d(3, 1, (1, 8), stride=1, bias=False)

        # Initialize the first fully connected layer.
        self.fc1 = nn.Linear(202, 512)

        # Initialize the second fully connected layer.
        self.fc2 = nn.Linear(512, 3)

    def forward(self, data):
        """
        Defines the forward pass of the network.

        Parameters:
        - data (torch.Tensor): The input data (images) to the network. Expected shape is (batch_size, 3, height, width),
          where 3 represents the RGB channels.

        Returns:
        - torch.Tensor: The log softmax probabilities of the input data belonging to each class. The shape of the output
          tensor is (batch_size, num_classes), where num_classes is 3 in this implementation.
        """

        # Apply the first convolutional layer.
        x = self.conv1(data)

        # Flatten the output for the linear layer.
        x = x.view(-1, 202)

        # Apply the first fully connected layer and ReLU activation function.
        x = self.fc1(x)
        x = F.relu(x)

        # Apply the second fully connected layer.
        x = self.fc2(x)

        # Apply the log softmax activation function on the output layer to obtain log probabilities.
        return F.log_softmax(x, dim=1)


class PreproNetwork(nn.Module):
    """
    A preprocessing neural network module designed for modifying the input data's dimensions and potentially applying convolutional transformations.

    The network is initialized with three convolutional layers intended for processing input data. However, the current implementation of the forward pass does not apply these layers but instead permutes and squeezes the input tensor. This might be a placeholder for a more complex preprocessing routine to be implemented later.

    Attributes:
    - conv1 (nn.Conv2d): First convolutional layer with 9 input channels, 18 output channels, kernel size of 3, stride of 1, padding of 1, and no bias.
    - conv2 (nn.Conv2d): Second convolutional layer with 18 input channels, 32 output channels, kernel size of 3, stride of 1, padding of 1, and no bias.
    - conv3 (nn.Conv2d): Third convolutional layer with 32 input channels, 1 output channel, kernel size of 3, stride of 1, padding of 1, and no bias.
    """

    def __init__(self):
        super().__init__()

        # Define the convolutional layers with specified configurations.
        self.conv1 = nn.Conv2d(9, 18, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(18, 32, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, data):
        """
        Performs the preprocessing operations on the input data.

        The current implementation only modifies the input tensor's dimensions through permutation and squeezing.
        Specifically, it permutes the dimensions to rearrange the channel position and then squeezes out one dimension.
        This implementation might be intended as a placeholder, with actual convolutional processing to be added later.

        Parameters:
        - data (torch.Tensor): The input data to be preprocessed, expected to have a shape that includes a batch size, channels, height, and width.

        Returns:
        - torch.Tensor: The preprocessed data after permuting and squeezing operations.
        """

        # Permute the dimensions of the input data to rearrange the channel position.
        x = data.permute(0, 2, 3, 1)
        # Squeeze out the specified dimension (in this case, the second dimension).
        x = x.squeeze(1)

        # Note: Convolutional layers defined in the class are not used in the current forward pass.
        return x


def detectFg(data_vid, batchsize, device, X, W, B, F_W, F_B, proddis, sumdis, netP, netC):
    """
    Detects the foreground in video data using deep learning models and arithmetic operations.

    This function processes the input video data in batches, applying a series of operations to
    each batch to classify pixels or regions as foreground or background. It leverages a preprocessing
    network, arithmetic operations, and a classification network to achieve this.

    Parameters:
    - data_vid (torch.Tensor): The input video data as a tensor, with shape (frames, channels, height, width).
    - batchsize (int): The size of each batch to process the video data.
    - device (torch.device): The device (CPU or GPU) to perform computations on.
    - X (object): An object or structure to hold intermediate data.
    - W, B (torch.Tensor): Tensors representing weights or parameters for the arithmetic operations on foreground.
    - F_W, F_B (function): Functions representing the arithmetic operations to be applied using W and B, respectively.
    - proddis, sumdis (function): Functions to compute the arithmetic distance for W and B.
    - netP (torch.nn.Module): A PyTorch model for preprocessing the input data.
    - netC (torch.nn.Module): A PyTorch model for classifying the processed data as foreground or background.

    Returns:
    - numpy.ndarray: An array containing the foreground detection labels for each frame of the input video data.

    The function processes the video data in specified batch sizes, applying preprocessing, arithmetic
    operations based on the provided functions and tensors, and finally classifying the output to detect
    foreground. The classification results for each batch are collected and returned as a NumPy array.
    """

    # Initialize an array to hold the labels (foreground/background) for each frame.
    re_labs = np.zeros(data_vid.shape[0])

    # Process the video data without computing gradients for efficiency.
    with torch.no_grad():
        # Iterate over the video data in batches.
        for i in range(round(data_vid.shape[0] / batchsize + 0.499999999999999999)):
            # Extract the current batch and move it to the specified device.
            data = data_vid[i * batchsize : (i + 1) * batchsize].to(device, dtype=torch.float32)

            # Apply the preprocessing network to the current batch.
            X._f = netP(data)

            # Perform the arithmetic operations for foreground and background detection.
            output_W = arithmeticDis(X, W, F_W, proddis)
            output_B = arithmeticDis(X, B, F_B, sumdis)

            # Combine the outputs from the arithmetic operations.
            output_DIS = output_W._f + output_B._f

            # Classify the combined output to detect foreground.
            output_labs = netC(output_DIS)

            # Store the classification results in the results array.
            re_labs[i * batchsize : (i + 1) * batchsize] = (
                output_labs.argmax(dim=1, keepdim=True).cpu().detach().squeeze()
            )

    return re_labs
