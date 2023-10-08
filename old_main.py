# -*- coding: utf-8 -*-
from tqdm import tqdm
from random import uniform
from models.decoder import Decoder
from models.encoder import Encoder
from models.discriminator import Discriminator
from models.generator import Generator
from readerDS import get_batch
import torch as t
from torch import nn, optim
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
import visdom
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train():

    channels = 3
    width = 64
    height = 64
    latent_dim = 100
    w_multi = 64
    lr =0.0002
    b1 =0.5
    b2 =0.999
    batch_size = 16

    vis = visdom.Visdom()

    train_loader = get_batch('/media/lightdi/CRUCIAL/Datasets/SSRC_Dataset/img/',
            '/media/lightdi/CRUCIAL/Datasets/SSRC_Dataset/img/Load_Prototipe.txt',
            batch_size)
    
    vis = visdom.Visdom()
    writer = SummaryWriter()

    G = Generator(latent_dim, channels, w_multi).cuda()
    C = Discriminator(w_multi, channels).cuda()
    E = Encoder(latent_dim, channels, w_multi).cuda()
    D = Decoder(latent_dim,channels, w_multi).cuda()

    G.train()
    C.train()
    E.train()
    D.train()
    
    loss_criterion_BCE = nn.BCEWithLogitsLoss()


    batch_ones_label = t.ones(batch_size).unsqueeze(1).cuda()
    batch_zeros_label = t.zeros(batch_size).unsqueeze(1).cuda()

    soft = t.nn.Softmax().cuda()

    optimizer_E = t.optim.Adam(E.parameters(),
                                    lr=lr, betas=(b1, b2))

    optimizer_D = t.optim.Adam(D.parameters(),
                                    lr=lr, betas=(b1, b2))

    optimizer_G = t.optim.Adam(G.parameters(),
                                    lr=lr, betas=(b1, b2))

    optimizer_C = t.optim.Adam(C.parameters(),
                                    lr=lr, betas=(b1, b2))

    init = 1
    epochs = 100
    for epoch in range(init, epochs+1): 
        print('%d epoch ...'%(epoch))
        g_loss = 0 

        pbar = tqdm(total=train_loader.__len__())
        for i, batch_data in enumerate(train_loader):
            # Iniciando
            G.zero_grad()
            C.zero_grad()
            E.zero_grad()
            D.zero_grad()

            batch_image = batch_data[0]
            batch_id_label = batch_data[1]
            batch_pro = batch_data[3]

            x_real = batch_image.cuda()

            eps = t.normal(0., 1., size=(batch_size, latent_dim)).cuda()
            xi = t.normal(0., 1., size=(batch_size, latent_dim)).cuda()

            z_real_mu, z_real_log_sigma = E(x_real)
            z_real = z_real_mu + t.exp(z_real_log_sigma) * eps
            x_real_mu = D(z_real)

            z_fake = t.normal(0., 1., size=(batch_size, latent_dim)).cuda()

            x_fake = G(t.cat((z_fake,xi),1))
            d_real = C(x_real)
            d_fake = C(x_fake)

            z_fake_mu, z_fake_log_sigma = E(x_fake)

            kl_loss =  t.mean(t.sum((z_real_mu**2 + t.exp(2*z_real_log_sigma))/2 - 1/2 - z_real_log_sigma, dim=1),-1)
            ll_loss = t.mean(t.sum(1/2 * t.square((x_real - x_real_mu) / 1) ,dim=(1,2,3)),-1)
            gen_loss = -d_fake
            critic_loss = (loss_criterion_BCE(batch_ones_label,d_real) +
                        loss_criterion_BCE(batch_zeros_label, d_fake)) 
            lat_loss = t.sum(
                        1/2* t.square(z_fake - z_fake_mu)*
                        (soft(-2*z_fake_log_sigma)).detach()
                        ) 

            encoder_loss = 1* kl_loss + ll_loss
            decoder_loss = ll_loss
            generator_loss = t.sum(gen_loss + lat_loss)

            #Encoder loss
            encoder_loss.backward()
            optimizer_E.step()

            #Decoder loss
            decoder_loss.backward()
            optimizer_D.step()

            #Critic loss
            critic_loss.backward()
            optimizer_C.step()

            #Generator loss
            generator_loss.backward()
            optimizer_G.step()
        
        x = vutils.make_grid(x_fake, normalize=True, scale_each=True)
        writer.add_image('Image', x, i)

        vis.images(x_fake/2+0.5,nrow=4,win='generated', opts={'title':"Generated"})
        vis.images(batch_image/2+0.5,nrow=4,win='original', opts={'title':"Original"})
        vis.images(batch_pro/2+0.5,nrow=4,win='prototipo', opts={'title':"Prototype"})


if __name__=='__main__':
    
    train()