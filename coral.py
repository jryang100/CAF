import torch


def CORAL(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]

    xm = torch.mean(source, 0, keepdim=True) - source
    xmt = torch.mean(target, 0, keepdim=True) - target

    xc = xm.t() @ xm / (ns - 1)
    xct = xmt.t() @ xmt / (nt - 1)

    loss = torch.norm(xc - xct, p='fro') ** 2 + 1e-8

    return loss
