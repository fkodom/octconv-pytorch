# OctConv-PyTorch

PyTorch implementation of Octave Convolution layers, as described in: [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049)


### Layers Contained

```
OctConv1d
OctConv2d
OctConv3d
OctConvTranspose1d
OctConvTranspose2d
OctConvTranspose3d
```

Currently, the following keyword arguments for convolution layers are not supported by the implemented OctConv layers:
```
stride
dilation
groups
```


### Example Usage

```python
import torch
from octconv import OctConv2d

batch, chin, nrow, ncol = 10, 16, 64, 64
chout, kernel_size, alpha_in, alpha_out = 32, 3, 0.25, 0.25

x = torch.randn((batch, chin, nrow, ncol))
layer = OctConv2d(chin, chout, kernel_size, alpha_in=alpha_in, alpha_out=alpha_out)
y = layer(x)
```