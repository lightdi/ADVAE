import torch
import torchvision as tvison
from torch import nn, optim

class Encoder(nn.Module):
    """Encoder

    Args:
        nn (_type_): _description_
    """

    def __init__(self, latent_dim, channel_num, w_multi):
        super(Encoder, self).__init__()

        self.W_multi = w_multi


        from_rgb = [
            
                 #ToRGB
                nn.Conv2d(channel_num, w_multi, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(w_multi,momentum=0.9),
                nn.LeakyReLU(),

        ]

        self.from_rgb = nn.Sequential(*from_rgb)

        disc1 = [
                        #DownBlock
                        nn.Conv2d(w_multi, w_multi*2,kernel_size=(4,4), padding=1, stride=2, bias=False,),
                        #modification
                        #nn.Dropout(0.3),
                        nn.BatchNorm2d(w_multi * 2,momentum=0.9),
                        nn.LeakyReLU(),
                    ]

        self.disc1 = nn.Sequential(*disc1)

        disc2 = [
                        #DownBlock
                        nn.Conv2d(w_multi*2, w_multi*4,kernel_size=(4,4), padding=1, stride=2, bias=False,),
                        #modification
                        #nn.Dropout(0.3),
                        nn.BatchNorm2d(w_multi * 4, momentum=0.9),
                        nn.LeakyReLU(),
        
                    ]

        self.disc2 = nn.Sequential(*disc2)

        disc3 = [

                        #DownBlock
                        nn.Conv2d(w_multi*4, w_multi*8,kernel_size=(4,4), padding=1, stride=2, bias=False,),
                        #modification
                        #nn.Dropout(0.3),
                        nn.BatchNorm2d(w_multi * 8, momentum=0.9),
                        nn.LeakyReLU(),
                        #modification
                        #nn.ReLU(),

                    ]

        self.disc3 = nn.Sequential(*disc3)

        self.first_norm = nn.BatchNorm2d(w_multi*8)

        self.reshape = nn.Flatten()

        self.dense_mu = nn.Linear(w_multi*8*4*4, latent_dim)

        self.log_sigma = nn.Linear(w_multi*8*4*4, latent_dim)

        for m in self.modules():
            if  isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                pass
                m.weight.data.normal_(0, 0.02)
            
            elif isinstance(m, nn.Linear):
                pass
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        
    def forward(self, input):
        x = self.from_rgb(input)
        x = self.disc1(x)
        x = self.disc2(x)
        x = self.disc3(x)
        x = self.reshape(x)
        mu = self.dense_mu(x)
        log_sigma = self.log_sigma(x)
        return mu, log_sigma