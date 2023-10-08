import torch
import torchvision
from scipy.linalg import sqrtm
import numpy as np

def calculate_fid_score(real_images, generated_images, batch_size, device):
    # Carregar uma rede neural pré-treinada para classificação de imagens
    inception_model = torchvision.models.inception_v3(pretrained=True)
    inception_model.to(device)
    inception_model.eval()
    inception_model.fc = torch.nn.Identity() # Remover a camada de saída

    # Extrair as características das imagens reais
    real_features = []
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size].to(device)
        with torch.no_grad():
            features = inception_model(batch).detach().cpu().numpy()
            real_features.append(features)
    real_features = np.concatenate(real_features, axis=0)

    # Extrair as características das imagens geradas
    generated_features = []
    for i in range(0, len(generated_images), batch_size):
        batch = generated_images[i:i+batch_size].to(device)
        with torch.no_grad():
            features = inception_model(batch).detach().cpu().numpy()
            generated_features.append(features)
    generated_features = np.concatenate(generated_features, axis=0)

    # Calcular a média e a matriz de covariância das características das imagens reais e geradas
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # Calcular a distância de Frechet entre as distribuições das características das imagens reais e geradas
    cov_sqrt = sqrtm(sigma_real.dot(sigma_generated))
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fid_score = np.sum((mu_real - mu_generated)**2) + np.trace(sigma_real + sigma_generated - 2*cov_sqrt)

    return fid_score