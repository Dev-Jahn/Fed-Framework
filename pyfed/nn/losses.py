from typing import Tuple, Union, Any

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def build_loss(name, *args, **kwargs):
    if name == 'ce':
        return CELossBase(*args, **kwargs)
    elif name == 'srip':
        return SRIP(*args, **kwargs)
    elif name == 'ocnn':
        return OCNN(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unimplemented loss "{name}"')


class CELossBase(nn.CrossEntropyLoss):
    """
    Base class for various type of CrossEntropy based losses
    Receives additional argument for forward function
    """

    def __init__(self, weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(CELossBase, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(input, target)


class KDLoss(CELossBase):
    """
    From paper, 'Distilling the Knowledge in a Neural Network'
    (Geoffry Hinton, Oriol Vinyals, Jeff Dean)
    (https://arxiv.org/pdf/1503.02531.pdf)
    """

    def __init__(self, *args, **kwargs):
        super(KDLoss, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor,
                soft_target: Tensor = None, lambda_: float = 0.5, temp: float = 1.0, *args, **kwargs) -> Tensor:
        assert soft_target is not None
        assert 0 <= lambda_ <= 1
        celoss = super().forward(input, target)
        kdloss = F.kl_div(
            F.log_softmax(input / temp, dim=1),
            F.softmax(soft_target / temp, dim=1),
            reduction='batchmean'
        ) * (temp ** 2)
        return celoss * (1 - lambda_) + kdloss * lambda_


class SRIP(CELossBase):
    """
    From paper, 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?'
    (Nitin Bansal, Xiaohan Chen, Zhangyang Wang)
    (https://arxiv.org/abs/1810.09102)
    """

    def __init__(self, *args, **kwargs):
        super(SRIP, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs):
        model = kwargs.get('model')
        decay = kwargs.get('decay')
        assert model and decay

        celoss = super().forward(input, target)
        oloss = self.l2_reg_ortho(model)
        return celoss + decay * oloss

    @staticmethod
    def l2_reg_ortho(model):
        device = next(model.parameters()).device
        l2_reg = None
        for W in model.parameters():
            if W.ndimension() < 2:
                continue
            else:
                cols = W[0].numel()
                rows = W.shape[0]
                w1 = W.view(-1, cols)
                wt = torch.transpose(w1, 0, 1)
                if rows > cols:
                    m = torch.matmul(wt, w1)
                    I = torch.eye(cols, cols, device=device)
                else:
                    m = torch.matmul(w1, wt)
                    I = torch.eye(rows, rows, device=device)
                w_tmp = (m - I)
                b_k = torch.rand(w_tmp.shape[1], 1, device=device)

                v1 = torch.matmul(w_tmp, b_k)
                norm1 = torch.norm(v1, 2)
                v2 = torch.div(v1, norm1)
                v3 = torch.matmul(w_tmp, v2)

                if l2_reg is None:
                    l2_reg = (torch.norm(v3, 2)) ** 2
                else:
                    l2_reg = l2_reg + (torch.norm(v3, 2)) ** 2
        return l2_reg


class OCNN(CELossBase):
    """
    Orthogonal Convolutional Neural Network Loss
    From paper 'Orthogonal Convolutional Neural Networks'
    (Jiayun Wang, Yubei Chen, Rudrasis Chakraborty, Stella X. Yu)
    (https://arxiv.org/abs/1911.12207)
    """

    def __init__(self, *args, **kwargs):
        super(OCNN, self).__init__(*args, **kwargs)
        self.device = None

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        model = kwargs.get('model')
        self.device = next(model.parameters()).device
        decay = kwargs.get('decay')
        assert model and decay

        celoss = super().forward(input, target)
        # TODO Model-wise tweak needed
        # 1x1 Conv
        diffs = [
            self.orth_dist(m.weight) for k, m in list(model.named_modules())
            if m.__dict__.get('kernel_size') == (1, 1)
        ]
        # Conv1 except first layer(according to original paper)
        diffs += [self.deconv_orth_dist(m.weight, stride=m.stride, padding=m.padding) for k, m in
                  list(model.named_modules()) if 'conv1' in k][1:]
        # Conv2 (Experimental)
        # diffs += [v for k, v in list(model.named_modules()) if 'conv2' in k]
        oloss = sum(diffs)
        return celoss + decay * oloss

    def orth_dist(self, mat, stride=None):
        mat = mat.reshape((mat.shape[0], -1))
        if mat.shape[0] < mat.shape[1]:
            mat = mat.permute(1, 0)
        return torch.norm(torch.t(mat) @ mat - torch.eye(mat.shape[1], device=self.device))

    def deconv_orth_dist(self, kernel, stride=2, padding=1):
        [o_c, i_c, w, h] = kernel.shape
        output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1]), device=self.device)
        ct = int(np.floor(output.shape[-1] / 2))
        target[:, :, ct, ct] = torch.eye(o_c, device=self.device)
        return torch.norm(output - target)

    def conv_orth_dist(self, kernel, stride=1):
        [o_c, i_c, w, h] = kernel.shape
        assert (w == h), "Do not support rectangular kernel"
        # half = np.floor(w/2)
        assert stride < w, "Please use matrix orthgonality instead"
        new_s = stride * (w - 1) + w  # np.int(2*(half+np.floor(half/stride))+1)
        temp = torch.eye(new_s * new_s * i_c, device=self.device).reshape((new_s * new_s * i_c, i_c, new_s, new_s))
        out = (F.conv2d(temp, kernel, stride=stride)).reshape((new_s * new_s * i_c, -1))
        Vmat = out[np.floor(new_s ** 2 / 2).astype(int)::new_s ** 2, :]
        temp = np.zeros((i_c, i_c * new_s ** 2))
        for i in range(temp.shape[0]):
            temp[i, np.floor(new_s ** 2 / 2).astype(int) + new_s ** 2 * i] = 1
        return torch.norm(Vmat @ torch.t(out) - torch.from_numpy(temp, ).float().to(self.device))


class PCC(CELossBase):
    """
    Principal Curvature Correction Loss
    """

    def __init__(self, *args, **kwargs):
        super(PCC, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError
