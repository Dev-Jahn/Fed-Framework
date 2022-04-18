import torch
from torch import nn, Tensor
import torch.nn.functional as F


def build_loss(name, *args, **kwargs):
    if name == 'ce':
        return nn.CrossEntropyLoss(*args, **kwargs)
    elif name == 'orth':
        return OrthLoss(*args, **kwargs)


# From paper 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?'
class OrthLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(OrthLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, model: nn.Module, decay: float, device) -> Tensor:
        celoss = F.cross_entropy(
            input, target,
            weight=self.weight, ignore_index=self.ignore_index,
            reduction=self.reduction
        )
        oloss = decay * self.l2_reg_ortho(model, device)
        return celoss + oloss

    @staticmethod
    def l2_reg_ortho(model, device):
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
                    I = torch.eye(cols, cols)
                else:
                    m = torch.matmul(w1, wt)
                    I = torch.eye(rows, rows)

                I = I.to(device)
                w_tmp = (m - I)
                b_k = torch.rand(w_tmp.shape[1], 1).to(device)

                v1 = torch.matmul(w_tmp, b_k)
                norm1 = torch.norm(v1, 2)
                v2 = torch.div(v1, norm1)
                v3 = torch.matmul(w_tmp, v2)

                if l2_reg is None:
                    l2_reg = (torch.norm(v3, 2)) ** 2
                else:
                    l2_reg = l2_reg + (torch.norm(v3, 2)) ** 2
        return l2_reg
