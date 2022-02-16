import torch


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
            if (rows > cols):
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
