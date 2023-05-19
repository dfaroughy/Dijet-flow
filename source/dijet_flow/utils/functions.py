import torch

#...preprocessing:

def logit(t, alpha=1e-6) -> torch.Tensor:
    x = alpha + (1 - 2 * alpha) * t
    return torch.log(x/(1-x))

def expit(t, alpha=1e-6) -> torch.Tensor:
    exp = torch.exp(t)
    x = exp / (1 + exp) 
    return (x - alpha) / (1 - 2*alpha)

