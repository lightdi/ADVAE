import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from scipy import linalg
import numpy as np


def torch_cov(m, rowvar=False):
    if rowvar:
        m = m.t()
    N = m.size()[0]
    m = m.unsqueeze(0)
    ones = torch.ones([N, 1], dtype=m.dtype, device=m.device)
    c = torch.matmul(m - torch.matmul(ones, m) / N, m.t() - torch.matmul(ones, m) / N) / (N - 1)
    return c


def calculate_activation_statistics(image, model):
    model.eval()
    device = torch.device("cuda")
    
    act = torch.tensor([]).to(device=device)

    with torch.no_grad():
        image1 = F.interpolate(image, size=(299, 299), mode='bilinear', align_corners=False)
        pred = model(image1)[0]
        #pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        act = torch.cat((act, pred), dim=0)

    mu = torch.mean(act, dim=0)
    sigma = torch_cov(act, rowvar=False)

    return mu.cpu().numpy(), sigma.cpu().numpy()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    fid_epsilon = 1e-6
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * fid_epsilon
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    # calculate distance between means
    diff = mu1 - mu2
    dist = (diff @ diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return np.real(dist)

def calculate_fid(real_image, gen_image):
    #Load the model
    inception_model = models.inception_v3(pretrained=True, aux_logits=True)
    device = torch.device("cuda")
    inception_model.to(device=device)
    inception_model.eval()
    
    mu1, sigma1 = calculate_activation_statistics(real_image, inception_model)
    mu2, sigma2 = calculate_activation_statistics(gen_image, inception_model)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid