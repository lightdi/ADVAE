import torch
from torchvision.models import inception_v3
import numpy as np


def get_fid(img_ref, img_samp):
    # Carrega o modelo Inception v3 pré-treinado
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model.to( torch.device("cuda"))

    # Define a camada para extrair as características
    # A camada de saída antes da classificação tem dimensão 2048
    dims = 2048
    inception_model.fc = torch.nn.Identity()

    # Carrega a imagem de referência e a imagem de amostra
    #img_ref =  torch.randn(1, 3, 64, 64)  # A imagem de referência é um tensor de tamanho (1, 3, 64, 64)
    #img_samp = torch.randn(1, 3, 64, 64) # A imagem de amostra é um tensor de tamanho (1, 3, 64, 64)

    # Redimensiona as imagens para (1, 3, 299, 299)
    img_ref_resized = torch.nn.functional.interpolate(img_ref, size=(299, 299), mode='bilinear', align_corners=False)
    img_samp_resized = torch.nn.functional.interpolate(img_samp, size=(299, 299), mode='bilinear', align_corners=False)

    # Extrai as características das imagens usando a camada definida
    with torch.no_grad():
        features_ref = inception_model(img_ref_resized).squeeze()
        features_samp = inception_model(img_samp_resized).squeeze()

    # Calcula a média das características das imagens
    mu_ref = torch.mean(features_ref, dim=0, keepdim=True)
    mu_samp = torch.mean(features_samp, dim=0, keepdim=True)

    # Calcula a matriz de covariância das características das imagens
    sigma_ref = torch.from_numpy(np.var(features_ref.cpu().numpy(), keepdims=True,ddof=1))
    sigma_samp = torch.from_numpy(np.var(features_samp.cpu().numpy(), keepdims=True,ddof=1))

    # Calcula a diferença entre as médias das características das imagens
    diff = mu_ref - mu_samp
    #diff = diff.cpu().numpy()
    # Calcula o FID
    matmul_result = torch.matmul(diff, torch.inverse(sigma_ref), diff.T)
    trace_result = torch.trace(sigma_ref + sigma_samp - 2 * torch.sqrt(torch.mm(sigma_ref, sigma_samp)))
    fid = float(matmul_result + trace_result)
    #fid = float(diff @ sigma_ref @ diff.T + torch.trace(sigma_ref + sigma_samp - 2 * torch.sqrt(sigma_ref @ sigma_samp)))
    return fid