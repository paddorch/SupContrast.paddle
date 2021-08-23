import paddle
import numpy as np
from paddle.vision.transforms import functional, transforms, BaseTransform

from .cutout import Cutout
from .autoaugment import CIFAR10Policy

class RandomGrayscale(BaseTransform):
    """Randomly convert image to grayscale with a probability of prob (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        prob (float): probability that image should be converted to grayscale.
        num_output_channels (int): (1 or 3) number of channels desired for output image
        keys (list[str]|tuple[str], optional): Same as ``BaseTransform``. Default: None.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """

    def __init__(self, prob=0.1, num_output_channels=1, keys=None):
        super().__init__()
        super(RandomGrayscale, self).__init__(keys)
        self.prob = prob
        self.num_output_channels = num_output_channels

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        if paddle.rand(1) < self.prob:
            return functional.to_grayscale(img, self.num_output_channels)
        return img


class RandomApply(object):
    """Random apply a transform"""
    def __init__(self, transform, prob=0.5):
        super().__init__()
        self.prob = prob
        self.transform = transform

    def __call__(self, img):
        if self.prob < np.random.rand():
            return img
        img = self.transform(img)
        return img


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
         # Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
         transforms.Normalize(cifar_mean, cifar_std)])

    # RandAugment
    # train_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop(size=32, scale=(0.3, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     RandomApply(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), prob=0.8),
        # RandomGrayscale(prob=0.2, num_output_channels=1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(cifar_mean, cifar_std),
    # ])

    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(cifar_mean, cifar_std)])
    return train_transforms, test_transforms