import paddle
import numpy as np

# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.half_length = length // 2

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.half_length, 0, h)
            y2 = np.clip(y + self.half_length, 0, h)
            x1 = np.clip(x - self.half_length, 0, w)
            x2 = np.clip(x + self.half_length, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = paddle.to_tensor(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img