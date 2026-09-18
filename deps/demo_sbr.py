"""Reusable SBR network and post-processing components for the LTS demo."""

import random

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from deps.common_py.evaluationBS import evaluation_numpy_entry_torch, normalize_mask


class OutConv(nn.Module):
    """
    A PyTorch module for applying a 2D convolution operation with a kernel size of 1.

    This module is typically used to change the number of channels in the input tensor
    without affecting its height and width, making it suitable for tasks like channel
    transformation at the end of a CNN architecture.

    Parameters:
    - in_channels (int): The number of channels in the input tensor.
    - out_channels (int): The desired number of channels in the output tensor.

    Attributes:
    - conv (nn.Conv2d): The convolutional layer that performs the actual channel
      transformation operation.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the OutConv module with a convolutional layer.

        Parameters:
        - in_channels (int): The number of channels in the input tensor.
        - out_channels (int): The desired number of channels in the output tensor.
        """
        super().__init__()  # Initialize the parent class (nn.Module)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # A convolutional layer with kernel_size=1 is used for channel-wise transformation.

    def forward(self, x):
        """
        Defines the forward pass of the module.

        Parameters:
        - x (Tensor): The input tensor with shape (batch_size, in_channels, H, W).

        Returns:
        - Tensor: The output tensor after applying the 2D convolution, with shape
          (batch_size, out_channels, H, W).
        """
        return self.conv(
            x
        )  # Apply the convolutional layer to the input tensor and return the result.


class DoubleConv(nn.Module):
    """
    A PyTorch module implementing a double convolutional block where each convolution
    operation is followed by a Batch Normalization (BN) and a ReLU activation function.
    This sequence is repeated twice.

    The block structure is: (Convolution => ReLU) * 2

    Parameters:
    - in_channels (int): Number of channels in the input tensor.
    - out_channels (int): Number of channels in the output tensor after the second convolution.
    - mid_channels (int, optional): Number of channels after the first convolution. If not
      specified, it defaults to the value of out_channels.

    Attributes:
    - double_conv (nn.Sequential): Sequential container of the two convolutional operations
      each followed by a ReLU activation function.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Initializes the DoubleConv module with the specified number of input, mid, and
        output channels. If mid_channels is not provided, it is set to the same value as
        out_channels by default.

        Parameters:
        - in_channels (int): Number of channels in the input tensor.
        - out_channels (int): Number of channels in the output tensor after the second convolution.
        - mid_channels (int, optional): Number of channels after the first convolution. Defaults to
          out_channels if not specified.
        """
        super().__init__()
        if not mid_channels:  # Check if mid_channels is provided, otherwise set it to out_channels
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        """
        Defines the forward pass of the DoubleConv module.

        Parameters:
        - x (Tensor): The input tensor with shape (batch_size, in_channels, H, W).

        Returns:
        - Tensor: The output tensor after processing through two convolutional operations,
          each followed by a ReLU activation, with shape (batch_size, out_channels, H, W).
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    A PyTorch module that performs downscaling on an input tensor using max pooling followed by
    a double convolution block. This module is typically used in CNN architectures to reduce the
    spatial dimensions of the input tensor while increasing the depth (number of channels),
    allowing the model to capture more complex features at a lower resolution.

    Parameters:
    - in_channels (int): Number of channels in the input tensor.
    - out_channels (int): Number of channels in the output tensor after the double convolution.

    Attributes:
    - maxpool_conv (nn.Sequential): A sequential container that first applies max pooling to
      downscale the input tensor and then performs a double convolution operation.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initializes the Down module with max pooling for downscaling followed by a double
        convolution block for feature extraction at the reduced resolution.

        Parameters:
        - in_channels (int): Number of channels in the input tensor.
        - out_channels (int): Number of channels in the output tensor after the double convolution.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Max pooling with a kernel size of 2 for downscaling
            DoubleConv(in_channels, out_channels),  # Double convolution block
        )

    def forward(self, x):
        """
        Defines the forward pass through the Down module.

        Parameters:
        - x (Tensor): The input tensor with shape (batch_size, in_channels, H, W).

        Returns:
        - Tensor: The output tensor after downscaling and double convolution, with reduced spatial
          dimensions and potentially increased number of channels, depending on out_channels. The
          shape is (batch_size, out_channels, H/2, W/2) assuming the input height and width are
          divisible by 2.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    A PyTorch module for upscaling the spatial dimensions of an input tensor followed by
    a double convolution operation. The module supports bilinear upsampling or transposed
    convolution for upscaling. This is typically used in the decoder part of segmentation
    networks to increase the resolution of feature maps.

    Parameters:
    - in_channels (int): Number of channels in the input tensor before upscaling.
    - out_channels (int): Number of channels in the output tensor after double convolution.
    - bilinear (bool): If True, use bilinear interpolation for upsampling. Otherwise,
      use transposed convolution. Defaults to True.
    - ext (bool): Experimental flag for altering input channels before double convolution.
      Defaults to False.

    Attributes:
    - up (nn.Module): The upsampling module, either nn.Upsample for bilinear interpolation
      or nn.ConvTranspose2d for transposed convolution.
    - conv (DoubleConv): The double convolution module applied after upsampling.
    """

    def __init__(self, in_channels, out_channels, bilinear=True, ext=False):
        super().__init__()

        # Upsampling strategy
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # For bilinear upsampling, adjust channels after upsampling
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # For transposed convolution, no channel adjustment needed before double conv
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            if ext:
                # Experimental feature to adjust in_channels
                in_channels = 20
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass for combining two feature maps with upscaling.

        Parameters:
        - x1 (Tensor): The tensor to be upscaled.
        - x2 (Tensor): The tensor to be concatenated with the upscaled x1.

        Returns:
        - Tensor: The result of upscaling x1, concatenating it with x2, and passing
          the combined tensor through a double convolution.
        """
        x1 = self.up(x1)  # Upscale x1

        # Ensure the dimensions match for concatenation
        if x1.shape[3] != x2.shape[3]:
            x1 = F.pad(x1, (0, 1))
        if x1.shape[2] != x2.shape[2]:
            x1 = F.pad(x1, (0, 0, 1, 0))

        x = torch.cat([x2, x1], dim=1)  # Concatenate x1 and x2 along the channel dimension
        return self.conv(x)  # Pass through double convolution


class U_Net(nn.Module):
    """
    Implements the U-Net architecture for semantic segmentation. U-Net is characterized by its
    encoder-decoder structure with skip connections, enabling precise localization. This class
    allows for customization of the number of channels in the input and output, as well as the
    choice between bilinear upsampling and transposed convolutions for upscaling in the decoder.

    Parameters:
    - n_channels (int): Number of channels in the input images.
    - n_classes (int): Number of output classes for segmentation.
    - bilinear (bool): If True, use bilinear upsampling for the decoder path. Otherwise, use
      transposed convolutions.

    Attributes:
    - Each encoder (down) and decoder (up) step is implemented as a separate module, allowing
      for modular composition of the network. The final layer is a convolution that maps the
      features to the prediction space of `n_classes`.
    """

    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Base number of channels for the U-Net architecture
        u_net_base_channels = 16

        # Encoder (downsampling path)
        self.unet_encode = DoubleConv(n_channels, u_net_base_channels)
        self.down1 = Down(u_net_base_channels, u_net_base_channels * 2)
        self.down2 = Down(u_net_base_channels * 2, u_net_base_channels * 4)
        self.down3 = Down(u_net_base_channels * 4, u_net_base_channels * 4)

        # Decoder (upsampling path)
        self.up1 = Up(u_net_base_channels * 8, u_net_base_channels * 2, bilinear)
        self.up2 = Up(u_net_base_channels * 4, u_net_base_channels, bilinear)
        self.up3 = Up(u_net_base_channels * 2, u_net_base_channels, bilinear)

        # Final output layer
        self.outc1 = OutConv(u_net_base_channels, n_classes)

    def forward(self, x):
        """
        Defines the forward pass through the U-Net model.

        Parameters:
        - x (Tensor): The input tensor of shape (batch_size, n_channels, H, W).

        Returns:
        - Tensor: The output segmentation map of shape (batch_size, n_classes, H, W).
        """

        # Encoder path
        x1 = self.unet_encode(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder path with skip connections
        x_u = self.up1(x4, x3)
        x_u = self.up2(x_u, x2)
        x_u = self.up3(x_u, x1)

        # Final convolution to get the segmentation map
        logits = self.outc1(x_u)
        return logits


def detect_current_level(
    img_patch_bags,
    gt_patch_bags,
    original_centers,
    network,
    len_batch,
    current_radius,
    original_gt,
    device=None,
):
    """
    Detects the current level's mask by processing image patches through a trained network
    and aggregating predictions to form a comprehensive mask aligned with the original
    image's dimensions.

    Parameters:
    - img_patch_bags (Tensor): A 4D tensor of image patches with shape [num_patches, 4, patch_size, patch_size].
    - gt_patch_bags (Tensor): A 4D tensor of ground truth patches, not used in the current implementation but
                              included for potential future extensions.
    - original_centers (list of lists): A 2D list containing the original center positions of each patch.
    - network (nn.Module): The trained network model for processing image patches.
    - len_batch (int): The batch size for processing patches through the network.
    - current_radius (int): The radius of the current level, used for adjusting the mask's placement.
    - original_gt (Tensor): The original ground truth 2D tensor, used for initializing the size of the refined mask.

    Returns:
    - refined_mask (Tensor): The aggregated and refined mask aligned with the original image's dimensions.
    - sampled_mask (Tensor): A mask indicating the sampling density across the original image's dimensions.
    """
    del gt_patch_bags
    NUM_DATA = img_patch_bags.shape[0]
    device = torch.device(device) if device is not None else img_patch_bags.device

    # Normalize image patches and transfer them to the model's device.
    imgs = img_patch_bags.to(device, dtype=torch.float32) / 255.0

    # Process the first batch outside the loop to initialize re_mask
    imgs_batch = imgs[:len_batch]
    with torch.no_grad():
        re_mask = network(imgs_batch)

    # Process remaining batches
    for batch_start in range(len_batch, NUM_DATA, len_batch):
        imgs_batch = imgs[batch_start : batch_start + len_batch]
        with torch.no_grad():
            mask = network(imgs_batch)
        re_mask = torch.cat((re_mask, mask), dim=0)

    re_mask = re_mask.argmax(dim=1, keepdim=True).squeeze()

    # Initialize buffers for refined and sampled masks on the same device.
    refined_mask = torch.zeros(original_gt.shape, dtype=torch.int64, device=device)
    sampled_mask = torch.zeros(original_gt.shape, dtype=torch.int64, device=device)

    # Padding to handle edge cases near borders
    pad = transforms.Pad(current_radius, padding_mode="reflect")
    refined_mask, sampled_mask = [
        pad(mask.unsqueeze(0)).squeeze(0) for mask in (refined_mask, sampled_mask)
    ]

    # Aggregate predictions over original image dimensions
    for idx, (x, y) in enumerate(original_centers):
        x, y = [pos + current_radius for pos in (x, y)]
        slice_ = (
            slice(x - current_radius, x + current_radius),
            slice(y - current_radius, y + current_radius),
        )
        refined_mask[slice_] += re_mask[idx]
        sampled_mask[slice_] += 1

    # Remove padding to match the original image dimensions
    refined_mask, sampled_mask = [
        mask[current_radius:-current_radius, current_radius:-current_radius]
        for mask in (refined_mask, sampled_mask)
    ]

    return refined_mask, sampled_mask


def cut_patches_randomly(imXfg, img, gt, num_pos, radius, device=None):
    """
    Extracts random patches from an image, its ground truth (GT), and a foreground indication tensor.

    Parameters:
    - imXfg (Tensor): A 2D tensor indicating the foreground areas of the image, of shape (row_im, column_im).
    - img (Tensor): The input image tensor of shape (row_im, column_im, channel).
    - gt (Tensor): The ground truth tensor for the image, of shape (row_im, column_im).
    - num_pos (int): The number of patches to extract.
    - radius (int): The radius of each patch. The diameter of patches will be 2*radius.

    Returns:
    - img_patch_bags (Tensor): A tensor containing the image patches of shape (num_pos, 3, 2*radius, 2*radius).
    - gt_patch_bags (Tensor): A tensor containing the GT patches of shape (num_pos, 2*radius, 2*radius).
    - imXfg_patch_bags (Tensor): A tensor containing the foreground patches of shape (num_pos, 2*radius, 2*radius).
    - original_centers (list): A list of the original center coordinates of each patch.
    """
    # Add an extra dimension to imXfg and gt and convert to float
    imXfg, gt = [x.unsqueeze(0).float() for x in (imXfg, gt)]

    # Reorder img to channel-first format and convert to float
    img = img.permute(2, 0, 1).float()

    # Apply padding to tensors
    pad = transforms.Pad(radius, padding_mode="reflect")
    img_pad, gt_pad, imXfg_pad = [pad(x) for x in (img, gt, imXfg)]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transfer tensors to the requested device as integer masks/images.
    img_pad, gt_pad, imXfg_pad = [x.int().to(device) for x in (img_pad, gt_pad, imXfg_pad)]

    # Remove the first dimension from gt_pad and imXfg_pad
    gt_pad, imXfg_pad = [x.squeeze(0) for x in (gt_pad, imXfg_pad)]

    # Initialize tensors for storing patches
    img_patch_bags = torch.zeros(
        (num_pos, 3, radius * 2, radius * 2), dtype=torch.int, device=device
    )
    gt_patch_bags = torch.zeros((num_pos, radius * 2, radius * 2), dtype=torch.int, device=device)
    imXfg_patch_bags = torch.zeros(
        (num_pos, radius * 2, radius * 2), dtype=torch.int, device=device
    )

    original_centers = []

    for i in range(num_pos):
        # Randomly choose a patch center within valid bounds
        center_x = random.randint(radius, img.shape[1] + radius - 1)
        center_y = random.randint(radius, img.shape[2] + radius - 1)

        # Extract the patch for img, gt, and imXfg
        slice_x = slice(center_x - radius, center_x + radius)
        slice_y = slice(center_y - radius, center_y + radius)
        img_patch_bags[i] = img_pad[:, slice_x, slice_y]
        gt_patch_bags[i] = gt_pad[slice_x, slice_y]
        imXfg_patch_bags[i] = imXfg_pad[slice_x, slice_y]

        # Adjust center coordinates to original image space
        original_centers.append([center_x - radius, center_y - radius])

    return img_patch_bags, gt_patch_bags, imXfg_patch_bags, original_centers


def detectFgImg_random(
    pa_im,
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
    radius=None,
):
    """
    Detects the foreground in an image using a network-based approach on randomly extracted patches across multiple scales.

    Parameters:
    - pa_im (str): Path to the directory containing the images.
    - ft_im (str): File type/extension of the images.
    - pa_fg (str): Path to the directory containing the foreground masks.
    - ft_fg (str): File type/extension of the foreground masks.
    - pa_gt (str): Path to the directory containing the ground truth masks.
    - ft_gt (str): File type/extension of the ground truth masks.
    - idx (int): Index of the image to process.
    - network (nn.Module): The neural network model for foreground detection.
    - device (str): The device to run the model on ('cuda' or 'cpu').
    - len_batch (int): Batch size for processing patches through the network.
    - layers (float): Scaling factor to adjust the number of positions based on the radius.
    - radius (list of int): Radii of patches to extract at different scales.

    Returns:
    - hist_mask (numpy.ndarray): Histogram of refined masks normalized by the sampled mask.
    - refinemask (numpy.ndarray): Binary mask obtained by applying a threshold to the hist_mask.
    - fg (Tensor): Foreground indication tensor.
    - gt (Tensor): Ground truth tensor.
    - refine_mask_different_level (list): List of refined masks at different levels.
    """
    if radius is None:
        radius = [16, 32, 64]

    # Construct file paths and read image and masks
    input_path, fg_path, gt_path = [
        f"{path}{prefix}{idx:06}.{ft}"
        for path, prefix, ft in zip(
            [pa_im, pa_fg, pa_gt], ["in", "gt", "gt"], [ft_im, ft_fg, ft_gt]
        )
    ]
    img = torch.from_numpy(imageio.imread(input_path))
    fg = normalize_mask(torch.from_numpy(imageio.imread(fg_path)))
    gt = normalize_mask(torch.from_numpy(imageio.imread(gt_path)))

    # Initialize refined and sampled masks
    refined_mask, sampled_mask = [
        torch.zeros(gt.shape, dtype=torch.int64, device=device) for _ in range(2)
    ]

    refine_mask_different_level = []

    # Process patches at different levels
    for current_radius in radius:
        current_num_pos = int(
            ((img.shape[0] * img.shape[1]) * layers) / ((2 * current_radius) ** 2)
        )

        img_patch_bags, gt_patch_bags, imXfg_patch_bags, original_centers = cut_patches_randomly(
            fg, img, gt, current_num_pos, current_radius, device=device
        )

        # Prepare patches for the network
        imXfg_patch_bags = imXfg_patch_bags.unsqueeze(1)
        img_patch_bags = torch.cat((img_patch_bags, imXfg_patch_bags), dim=1).to(device)

        # Detect foreground at current level
        current_refined_mask, current_sampled_mask = detect_current_level(
            img_patch_bags,
            gt_patch_bags,
            original_centers,
            network,
            len_batch,
            current_radius,
            gt,
            device=device,
        )

        # Accumulate results
        refined_mask += current_refined_mask
        sampled_mask += current_sampled_mask
        refine_mask_different_level.append(current_refined_mask.cpu().numpy())

    # Normalize and threshold the aggregated mask
    hist_mask = (refined_mask / sampled_mask.clamp(min=1)).cpu().numpy()  # Avoid division by zero
    hist_mask[np.isnan(hist_mask)] = 0  # Handle NaNs
    refinemask = (
        hist_mask > 0.5
    )  # Simple thresholding, can be replaced with Otsu's method if needed

    return hist_mask, refinemask, fg.cpu(), gt.cpu(), refine_mask_different_level


def update_metrics(fgim, gtim, rfim, metrics):
    """
    Updates the provided metrics dictionary with True Positives (TP), False Positives (FP),
    True Negatives (TN), and False Negatives (FN) based on the foreground image (fgim),
    ground truth image (gtim), and refined image (rfim).
    """
    TP, FP, TN, FN = evaluation_numpy_entry_torch(fgim, gtim)
    metrics["TP"] += TP
    metrics["FP"] += FP
    metrics["TN"] += TN
    metrics["FN"] += FN

    TP_rf, FP_rf, TN_rf, FN_rf = evaluation_numpy_entry_torch(rfim, gtim)
    metrics["TP_rf"] += TP_rf
    metrics["FP_rf"] += FP_rf
    metrics["TN_rf"] += TN_rf
    metrics["FN_rf"] += FN_rf


def print_cumulative_metrics(metrics):
    """
    Prints the cumulative Recall (Re), Precision (Pr), and F-measure (Fm) for both the original and refined foreground detection.
    """
    Re_sum = metrics["TP"] / max((metrics["TP"] + metrics["FN"]), 1)
    Pr_sum = metrics["TP"] / max((metrics["TP"] + metrics["FP"]), 1)
    Fm_sum = (2 * Pr_sum * Re_sum) / max((Pr_sum + Re_sum), 0.0001)

    print(f"\nAccumulate original: Re_sum: {Re_sum:4f}, Pr_sum: {Pr_sum:4f}, Fm_sum: {Fm_sum:4f}")

    Re_sum_rf = metrics["TP_rf"] / max((metrics["TP_rf"] + metrics["FN_rf"]), 1)
    Pr_sum_rf = metrics["TP_rf"] / max((metrics["TP_rf"] + metrics["FP_rf"]), 1)
    Fm_sum_rf = (2 * Pr_sum_rf * Re_sum_rf) / max((Pr_sum_rf + Re_sum_rf), 0.0001)

    print(
        f"Accumulate refinefg: Re_sum_rf: {Re_sum_rf:4f}, Pr_sum_rf: {Pr_sum_rf:4f}, Fm_sum_rf: {Fm_sum_rf:4f}"
    )


def display_images(imXfg, hist_mask, refinemask, gt):
    """
    Display the foreground, histogram mask, refined mask, and ground truth images side by side.

    Parameters:
    - imXfg (Tensor): The foreground image.
    - hist_mask (Tensor): The histogram mask.
    - refinemask (Tensor): The refined mask.
    - gt (Tensor): The ground truth mask.
    """
    _fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(imXfg.cpu().numpy(), cmap="gray")
    axs[0].set_title("DIDL Mask")
    axs[0].axis("off")

    axs[1].imshow(hist_mask.cpu().numpy(), cmap="gray")
    axs[1].set_title("Histogram Mask")
    axs[1].axis("off")

    axs[2].imshow(refinemask.cpu().numpy(), cmap="gray")
    axs[2].set_title("Refined Mask")
    axs[2].axis("off")

    axs[3].imshow(gt.cpu().numpy(), cmap="gray")
    axs[3].set_title("Ground Truth")
    axs[3].axis("off")

    plt.show()
