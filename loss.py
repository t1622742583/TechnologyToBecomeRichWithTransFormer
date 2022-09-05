import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target))
        bin_idx = self._g2bin(g)
        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = x.size(0)

        nonempty_bins = (bin_count > 0).sum().item()
        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd
        return self._custom_loss(x, target, beta[bin_idx])
class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return torch.sum(
            (torch.nn.NLLLoss(reduce=False)(torch.log(x), target)).mul(weight.to(device).detach())) / torch.sum(
            weight.to(device).detach())

    def _custom_loss_grad(self, x, target):
        x = x.cpu().detach()
        target = target.cpu()
        return torch.tensor([x[i, target[i]] for i in range(target.shape[0])]) - target