import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.dygraph import ParallelEnv
from paddle.fluid.layers.utils import flatten
from paddle.hapi.model import to_list

from .resnet import resnet18, resnet34, resnet50, resnet101

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class SupConResNet(nn.Layer):
    """backbone + projection head"""

    def __init__(self, name='resnet18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()

        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(num_classes=0)
        assert self.encoder.num_classes == 0  # resnet without classifier

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x).squeeze()
        feat = F.normalize(self.head(feat), axis=1)
        return feat


class SupConLoss(nn.Layer):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz * 2, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        batch_size: int = features.shape[0] // 2
        f1, f2 = paddle.split(features, 2, axis=0)
        features = paddle.concat([paddle.unsqueeze(f1, 1), paddle.unsqueeze(f2, 1)], axis=1)  # (bsz, n_views, ...)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:  # flatten features
            features = features.view(features.shape[0], features.shape[1], -1)  # (bsz, n_views, hidden_size)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = paddle.eye(batch_size, dtype=paddle.float32)
        elif labels is not None:
            labels = labels.reshape([-1, 1])
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = paddle.cast(paddle.equal(labels, labels.t()), dtype=paddle.float32)  # [bsz, bsz]
        else:
            mask = paddle.cast(mask, dtype=paddle.float32)

        contrast_count = features.shape[1]  # n_views: 2
        contrast_feature = paddle.concat(paddle.unbind(features, axis=1), axis=0)  # (2 * bsz, hidden_size)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # (2 * bsz, hidden_size)
            anchor_count = contrast_count  # n_views: 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits (2 * bsz, 2 * bsz)
        anchor_dot_contrast = paddle.divide(paddle.matmul(anchor_feature, contrast_feature.t()),
                                            paddle.to_tensor(self.temperature, dtype=paddle.float32))

        # for numerical stability: every row - row_max
        logits_max = paddle.max(anchor_dot_contrast, axis=1, keepdim=True)  # (2 * bsz, 1)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask (bsz, bsz) -> (anchor_count * bsz, contrast_count * bsz)
        mask = mask.tile((anchor_count, contrast_count))
        """ old version
        _mask = mask
        for _ in range(anchor_count - 1):
            mask = paddle.concat([mask, _mask], axis=0)
        _mask = mask
        for _ in range(contrast_count - 1):
            mask = paddle.concat([mask, _mask], axis=1)
        """

        # mask-out self-contrast cases
        logits_mask = paddle.ones_like(mask) - paddle.eye(anchor_count * batch_size, contrast_count * batch_size)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = paddle.exp(logits) * logits_mask
        log_prob = logits - paddle.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# _parallel_context_initialized = False


class SupConModel(paddle.Model):
    """Modified from paddle.Model._run_one_epoch"""

    def _run_one_epoch(self, data_loader, callbacks, mode, logs={}):
        outputs = []
        for step, data in enumerate(data_loader):
            data[0] = paddle.concat([data[0][0], data[0][1]], axis=0)  # the only line added

            data = flatten(data)

            batch_size = data[0].shape()[0] if callable(data[
                                                            0].shape) else data[0].shape[0]

            callbacks.on_batch_begin(mode, step, logs)

            if mode != 'predict':
                outs = getattr(self, mode + '_batch')(data[:len(self._inputs)],
                                                      data[len(self._inputs):])
                if self._metrics and self._loss:
                    metrics = [[l[0] for l in outs[0]]]
                elif self._loss:
                    metrics = [[l[0] for l in outs]]
                else:
                    metrics = []

                # metrics
                for metric in self._metrics:
                    res = metric.accumulate()
                    metrics.extend(to_list(res))

                assert len(self._metrics_name()) == len(metrics)
                for k, v in zip(self._metrics_name(), metrics):
                    logs[k] = v
            else:
                if self._inputs is not None:
                    outs = self.predict_batch(data[:len(self._inputs)])
                else:
                    outs = self.predict_batch(data)

                outputs.append(outs)

            logs['step'] = step
            if mode == 'train' or self._adapter._merge_count.get(
                    mode + '_batch', 0) <= 0:
                logs['batch_size'] = batch_size * ParallelEnv().nranks
            else:
                logs['batch_size'] = self._adapter._merge_count[mode + '_batch']

            callbacks.on_batch_end(mode, step, logs)
        self._reset_metrics()

        if mode == 'predict':
            return logs, outputs
        return logs
