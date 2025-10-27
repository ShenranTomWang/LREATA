import torch

class GaussianAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, miu: int = 0, sigma: int = 1):
        self.miu = miu
        self.sigma = sigma
        super().__init__(optimizer, 0)

    def get_lr(self):
        step = self.last_epoch
        factor = torch.exp(-0.5 * ((step - self.miu) / self.sigma) ** 2)
        return [base_lr * factor.item() for base_lr in self.base_lrs]

def get_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif name == "gaussian_annealing":
        return GaussianAnnealingLR(optimizer, **kwargs)