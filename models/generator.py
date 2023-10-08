import torch
import torchvision as tvison
from torch import nn, optim


class Generator(nn.Module):
    """

         Generator    
    
    """
    def __init__(self, latent_dim, channel_num, w_mult, noise_dim=50):
        super(Generator, self).__init__()
        self.w_mult = w_mult
        #Transform 128x128 to 96x96 image

        gen1 = [
            
                #UpBlock
                nn.ConvTranspose2d(w_mult*8, w_mult*4, kernel_size=(4,4), stride=2, bias=False, padding=1),
                
                nn.InstanceNorm2d(w_mult, affine=False, momentum=0.9),
                #nn.BatchNorm2d(w_mult*4, momentum=0.9),
                nn.ReLU(),
                #Modification
                #nn.SELU(),
        ]

        self.gen1 = nn.Sequential(*gen1)
        
        gen2 = [
            
                #UpBlock
                nn.ConvTranspose2d(w_mult*4, w_mult*2, kernel_size=(4,4), stride=2, bias=False, padding=1  ),
                nn.Dropout(0.1),
                nn.InstanceNorm2d(w_mult, affine=False, momentum=0.9),
                #nn.BatchNorm2d(w_mult*2, momentum=0.9),
                nn.ReLU(),
                #Modification
                #nn.SELU(),
        ]

        self.gen2 = nn.Sequential(*gen2)

        gen3 = [
            
                #UpBlock
                nn.ConvTranspose2d(w_mult*2, w_mult, kernel_size=(4,4), stride=2, bias=False, padding=1),
                nn.InstanceNorm2d(w_mult, affine=False, momentum=0.9),
                #nn.BatchNorm2d(w_mult, momentum=0.9),
                nn.ReLU(),
                #Modification
                #nn.SELU(),
        ]

        self.gen3 = nn.Sequential(*gen3)

        to_rgb = [
            
                 #ToRGB
                nn.ConvTranspose2d(w_mult, channel_num, kernel_size=4, stride=2, padding=1),
                #nn.Dropout(0.2),
                #Finlize
                nn.Tanh()
        ]

        self.to_rgb = nn.Sequential(*to_rgb)

        # get a Latent_dim(100) and transform into w_mult(64) * 128
        self.gen_fc = nn.Linear(latent_dim + noise_dim, w_mult*8*4*4, bias=False)
        # get result of Linear layer(8192) transform into (4,4,w_mult(64))
        # x = x.view(4, 4, w_mult*8)
        # First batch normalization
        self.first_norm = nn.BatchNorm2d(w_mult*8)
        # first_norm(x)
        self.activation = nn.ReLU()
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

            #elif isinstance(m, nn.InstanceNorm2d):
                #nn.init.normal_(m.weight.data, 1.0, 0.02)
                #nn.init.constant_(m.bias.data, 0)

    def forward(self,input):
        x = self.gen_fc(input)
        x = x.view(-1, self.w_mult*8,  4, 4)
        x = self.first_norm(x)
        x = self.activation(x)
        #print(x.shape)
        x = self.gen1(x)
        #print(x.shape)
        x = self.gen2(x)
        #print(x.shape)
        x = self.gen3(x)
        #print(x.shape)
        x = self.to_rgb(x)
        #print(x.shape)
        #x = self.gen_convLayers(x)

        return x
    


        """ gen_convLayers = [
            
                #UpBlock
                nn.ConvTranspose2d(w_mult*4, w_mult*2, kernel_size=5, bias=False, padding=0),
                nn.BatchNorm2d(w_mult*2, momentum=0.9),
                nn.ReLU(),

                #UpBlock
                nn.ConvTranspose2d(w_mult*2, w_mult, kernel_size=(5,5), bias=False, padding=1 ),
                nn.BatchNorm2d(w_mult, momentum=0.9),
                nn.ReLU(),

                #UpBlock
                nn.ConvTranspose2d(w_mult, w_mult, kernel_size=(5,5), bias=False, padding=1),
                nn.BatchNorm2d(w_mult, momentum=0.9),
                nn.ReLU(),

                #ToRGB
                nn.ConvTranspose2d(w_mult, channel_num, 5),

                #Finlize
                nn.Tanh()
            
            ]
        
        self.gen_convLayers = nn.Sequential(*gen_convLayers)

        #DownBlock
        #nn.Conv2d(g_in, g_out,kernel_size=(5,5),padding='SAME', bias=False,),
        #nn.BatchNorm2d(g_out,momentum=0.9),
        #nn.LeakyReLU(),
        """