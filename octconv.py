r"""
octconv.py
------------
Implementations of Octave Convolutions (as well as Transposed Convolutions), following the paper:
'Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution', Chen et. al. (2019)
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class OctConv_(nn.Module):
    r"""Generic parent class for all Octave Convolution modules.  Implements repetitive initialization statements."""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        super(OctConv_, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.padding = kernel_size // 2

        # Require the fractions of H/L filters to be in the range: [0, 1]
        assert 0 <= alpha_in <= 1
        assert 0 <= alpha_out <= 1

        # Define the numbers of input and output H/L channels
        self.channels_in_l = int(alpha_in * channels_in)
        self.channels_in_h = channels_in - self.channels_in_l
        self.channels_out_l = int(alpha_out * channels_out)
        self.channels_out_h = channels_out - self.channels_out_l

        # Initialize other class properties that will be defined for each module
        self.whh, self.whl, self.wlh, self.wll = None, None, None, None
        self.bias = None

    def get_pool_function_(self):
        r"""Returns a callable function, which implements average pooling for this Octave Convolution module
        """
        msg = f'Pooling function has not been defined for {self.__class__.__name__}.'
        raise NotImplementedError(msg)

    def get_conv_function_(self):
        r"""Returns a callable function, which implements convolution for this Octave Convolution module
        """
        msg = f'Convolution function has not been defined for {self.__class__.__name__}.'
        raise NotImplementedError(msg)

    def forward(self, x: Tensor) -> Tensor:
        r"""Pushes input data through the OctConv1d module.  Setting alpha_in=0, alpha_out=0 makes the OctConv1d
        equivalent to the PyTorch Conv1d module.

        :param x: Input tensor.  Shape: (batch, nchan, nrow, ncol)
        :return: Module outputs
        """
        batch = x.shape[0]
        fsize = x.shape[2:]
        # batch, _, seq_len = x.shape

        # Grab the convolution and pooling functions for rank = self.rank
        pool = self.get_pool_function_()
        conv = self.get_conv_function_()

        # Separate H, L channels, and downsample the L channels
        in_h = x[:, :self.channels_in_h]
        in_l = pool(x[:, self.channels_in_h:], 2) if self.channels_in_l > 0 else torch.empty(0)

        # It is possible that one or more of the output arrays will be empty (whenever alpha is 0 or 1).  This causes
        # an exception for the convolution function, so we need to check for it.  The corresponding weights
        # matrix will have size zero whenever the output is empty.
        out_hh = conv(in_h, self.whh, padding=self.padding) if self.whh.numel() else 0
        out_hl = conv(in_h, self.whl, padding=self.padding) if self.whl.numel() else 0
        out_lh = conv(in_l, self.wlh, padding=self.padding) if self.wlh.numel() else torch.empty(0)
        out_ll = conv(in_l, self.wll, padding=self.padding) if self.wll.numel() else torch.empty(0)
        out_lh = F.interpolate(out_lh, scale_factor=2, mode='nearest') if out_lh.numel() else 0
        out_ll = F.interpolate(out_ll, scale_factor=2, mode='nearest') if out_ll.numel() else 0

        # Need to cast each of the output arrays, in order to handle cases with empty arrays (discussed above).
        device = x.device.type
        outputs_h = torch.add(out_hh, out_lh) * torch.ones((batch, self.channels_out_h, *fsize), device=device)
        outputs_l = torch.add(out_hl, out_ll) * torch.ones((batch, self.channels_out_l, *fsize), device=device)
        outputs = torch.cat((outputs_h, outputs_l), 1)
        if self.bias is not None:
            outputs = outputs + self.bias

        return outputs


class OctConvNd(OctConv_):
    r"""Generic parent class for all non-transposed octave convolution modules.  Initializes weight, bias tensors."""

    def __init__(self, rank: int, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param rank: Number of dimensions for the convolutional kernel.  (e.g. 3D convolution --> rank = 3)
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        super(OctConvNd, self).__init__(channels_in, channels_out, kernel_size, alpha_in, alpha_out)
        self.rank = rank
        assert 0 < self.rank < 4

        ksize = tuple([kernel_size] * rank)
        self.whh = nn.Parameter(torch.zeros((self.channels_out_h, self.channels_in_h) + ksize))
        self.whl = nn.Parameter(torch.zeros((self.channels_out_l, self.channels_in_h) + ksize))
        self.wlh = nn.Parameter(torch.zeros((self.channels_out_h, self.channels_in_l) + ksize))
        self.wll = nn.Parameter(torch.zeros((self.channels_out_l, self.channels_in_l) + ksize))
        nn.init.kaiming_uniform_(self.whh, a=2) if self.whh.numel() else None
        nn.init.kaiming_uniform_(self.whl, a=2) if self.whl.numel() else None
        nn.init.kaiming_uniform_(self.wlh, a=2) if self.wlh.numel() else None
        nn.init.kaiming_uniform_(self.wll, a=2) if self.wll.numel() else None

        bias_size = (1, self.channels_out) + tuple([1] * self.rank)
        self.bias = nn.Parameter(torch.zeros(bias_size))


class OctConvTransposeNd(OctConv_):
    r"""Generic parent class for all transposed octave convolution modules.  Initializes weight, bias tensors."""

    def __init__(self, rank: int, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param rank: Number of dimensions for the convolutional kernel.  (e.g. 3D convolution --> rank = 3)
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        super(OctConvTransposeNd, self).__init__(channels_in, channels_out, kernel_size, alpha_in, alpha_out)
        self.rank = rank
        assert 0 < self.rank < 4

        ksize = tuple([kernel_size] * rank)
        self.whh = nn.Parameter(torch.zeros((self.channels_in_h, self.channels_out_h) + ksize))
        self.whl = nn.Parameter(torch.zeros((self.channels_in_h, self.channels_out_l) + ksize))
        self.wlh = nn.Parameter(torch.zeros((self.channels_in_l, self.channels_out_h) + ksize))
        self.wll = nn.Parameter(torch.zeros((self.channels_in_l, self.channels_out_l) + ksize))
        nn.init.kaiming_uniform_(self.whh, a=2) if self.whh.numel() else None
        nn.init.kaiming_uniform_(self.whl, a=2) if self.whl.numel() else None
        nn.init.kaiming_uniform_(self.wlh, a=2) if self.wlh.numel() else None
        nn.init.kaiming_uniform_(self.wll, a=2) if self.wll.numel() else None


class OctConv1d(OctConvNd):
    r"""PyTorch implementation of a 1D Octave Convolution, as in:
    'Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution', Chen et. al. (2019)"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        rank = 1
        super(OctConv1d, self).__init__(rank, channels_in, channels_out, kernel_size, alpha_in, alpha_out)

    def get_pool_function_(self):
        r"""Returns a callable function, which implements average pooling for this Octave Convolution module
        """
        return F.avg_pool1d

    def get_conv_function_(self):
        r"""Returns a callable function, which implements convolution for this Octave Convolution module
        """
        return F.conv1d


class OctConv2d(OctConvNd):
    r"""PyTorch implementation of a 2D Octave Convolution, as in:
     'Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution', Chen et. al. (2019)"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        rank = 2
        super(OctConv2d, self).__init__(rank, channels_in, channels_out, kernel_size, alpha_in, alpha_out)

    def get_pool_function_(self):
        r"""Returns a callable function, which implements average pooling for this Octave Convolution module
        """
        return F.avg_pool2d

    def get_conv_function_(self):
        r"""Returns a callable function, which implements convolution for this Octave Convolution module
        """
        return F.conv2d


class OctConv3d(OctConvNd):
    r"""PyTorch implementation of a 3D Octave Convolution, as in:
    'Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution', Chen et. al. (2019)"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        rank = 3
        super(OctConv3d, self).__init__(rank, channels_in, channels_out, kernel_size, alpha_in, alpha_out)

    def get_pool_function_(self):
        r"""Returns a callable function, which implements average pooling for this Octave Convolution module
        """
        return F.avg_pool3d

    def get_conv_function_(self):
        r"""Returns a callable function, which implements convolution for this Octave Convolution module
        """
        return F.conv3d


class OctConvTranspose1d(OctConvTransposeNd):
    r"""PyTorch implementation of a 1D Octave Convolution, as in:
    'Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution', Chen et. al. (2019)"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        rank = 1
        super(OctConvTranspose1d, self).__init__(rank, channels_in, channels_out, kernel_size, alpha_in, alpha_out)

    def get_pool_function_(self):
        r"""Returns a callable function, which implements average pooling for this Octave Convolution module
        """
        return F.avg_pool1d

    def get_conv_function_(self):
        r"""Returns a callable function, which implements convolution for this Octave Convolution module
        """
        return F.conv_transpose1d


class OctConvTranspose2d(OctConvTransposeNd):
    r"""PyTorch implementation of a 1D Octave Convolution, as in:
    'Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution', Chen et. al. (2019)"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        rank = 2
        super(OctConvTranspose2d, self).__init__(rank, channels_in, channels_out, kernel_size, alpha_in, alpha_out)

    def get_pool_function_(self):
        r"""Returns a callable function, which implements average pooling for this Octave Convolution module
        """
        return F.avg_pool2d

    def get_conv_function_(self):
        r"""Returns a callable function, which implements convolution for this Octave Convolution module
        """
        return F.conv_transpose2d


class OctConvTranspose3d(OctConvTransposeNd):
    r"""PyTorch implementation of a 1D Octave Convolution, as in:
    'Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution', Chen et. al. (2019)"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, alpha_in: float = 0.5,
                 alpha_out: float = 0.5):
        r"""
        :param channels_in: Number of input channels (features) to the module
        :param channels_out: Number of output channels (features) to the module
        :param kernel_size: Size of the convolutional kernels (int)
        :param alpha_in: Fraction of the input channels that are **low** frequency
        :param alpha_out:Fraction of the output channels that are **low** frequency
        """
        rank = 3
        super(OctConvTranspose3d, self).__init__(rank, channels_in, channels_out, kernel_size, alpha_in, alpha_out)

    def get_pool_function_(self):
        r"""Returns a callable function, which implements average pooling for this Octave Convolution module
        """
        return F.avg_pool3d

    def get_conv_function_(self):
        r"""Returns a callable function, which implements convolution for this Octave Convolution module
        """
        return F.conv_transpose3d
