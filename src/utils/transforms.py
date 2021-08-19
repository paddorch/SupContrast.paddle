import numpy as np
from paddle.vision.transforms import transforms

from .cutout import Cutout
from .autoaugment import CIFAR10Policy

# class ToArray(object):
#     """Convert a ``PIL.Image`` to ``numpy.ndarray``.
#     Converts a PIL.Image or numpy.ndarray (H x W x C) to a paddle.Tensor of shape (C x H x W).
#     If input is a grayscale image (H x W), it will be converted to a image of shape (H x W x 1).
#     And the shape of output tensor will be (1 x H x W).
#     If you want to keep the shape of output tensor as (H x W x C), you can set data_format = ``HWC`` .
#     Converts a PIL.Image or numpy.ndarray in the range [0, 255] to a paddle.Tensor in the
#     range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr,
#     RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8.
#     In the other cases, tensors are returned without scaling.
#     """
#     def __call__(self, img):
#         img = np.array(img)
#         img = np.transpose(img, [2, 0, 1])
#         img = img / 255.
#         return img.astype('float32')
#
#
# class RandomApply(object):
#     """Random apply a transform"""
#     def __init__(self, transform, p=0.5):
#         super().__init__()
#         self.p = p
#         self.transform = transform
#
#     def __call__(self, img):
#         if self.p < np.random.rand():
#             return img
#         img = self.transform(img)
#         return img

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def build_transform():
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.247, 0.243, 0.261]
    # AutoAugment
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(32, padding=4, fill=128),
         transforms.RandomHorizontalFlip(), CIFAR10Policy(),
         transforms.ToTensor(),
         Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
         transforms.Normalize(cifar_mean, cifar_std)])
    # RandAugment
    # train_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop(size=32, scale=(0.3, 1.)),
    #     RandomApply(
    #         transforms.ContrastTransform(0.1),
    #     ),
    #     RandomApply(
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #     , p=0.8),
    #     RandomApply(
    #         transforms.BrightnessTransform(0.1),
    #     ),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     ToArray(),
    #     transforms.Normalize(cifar_mean, cifar_std),
    # ])
    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(cifar_mean, cifar_std)])
    return train_transforms, test_transforms