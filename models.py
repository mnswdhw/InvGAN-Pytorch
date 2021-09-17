import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU


# all these architecures are as defined in the defenseGAN paper.

class MnistCnn(nn.Module):
    def __init__(self,):
        super(MnistCnn,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels = 1,out_channels = 64, kernel_size = 5, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64,out_channels = 64, kernel_size = 5, stride = 2),
            nn.ReLU(),
            nn.Dropout(p=0.25)

        )

        self.dense = nn.Sequential(
            nn.Linear(6400,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,10)
        )

    def forward(self,x):
        # x= input tensor 
        output = self.conv(x)
        # output = output.view(output.shape[0],-1)
        output = torch.reshape(output,(output.shape[0],output.shape[1]*output.shape[2]*output.shape[3]))
        output = self.dense(output)
        # output is now a tensor of shape (N,10) this is pre-softmax logits of N images
        soft = nn.Softmax(dim=1)
        output = soft(output)
        return output 

    def get_logits(self,x):

        output = self.conv(x)
        # output = output.view(output.shape[0],-1)
        output = torch.reshape(output,(output.shape[0],output.shape[1]*output.shape[2]*output.shape[3]))
        output = self.dense(output)
        logits = output
        return logits



class Model_b(nn.Module):
    def __init__(self):
        super(Model_b,self).__init__()

        self.conv = nn.Sequential(

            nn.Dropout(0.2),
            nn.Conv2d(1,64,8,2,padding = 3),
            nn.ReLU(),
            nn.Conv2d(64,128,6,2),
            nn.ReLU(),
            nn.Conv2d(128,128,5,1),
            nn.ReLU(),
            nn.Dropout(0.5)

        )

        self.linear = nn.Sequential(
            nn.Linear(128,10),
        )

    def forward(self,x):
        output = self.conv(x)
        # output = output.view(output.shape[0],-1)
        output = torch.reshape(output,(output.shape[0],output.shape[1]*output.shape[2]*output.shape[3]))
        output = self.linear(output)
        # output is now a tensor of shape (N,10) this is pre-softmax logits of N images
        soft = nn.Softmax(dim=1)
        output = soft(output)
        return output   

    def get_logits(self,x):

        output = self.conv(x)
        # output = output.view(output.shape[0],-1)
        output = torch.reshape(output,(output.shape[0],output.shape[1]*output.shape[2]*output.shape[3]))
        output = self.linear(output)
        logits = output
        return logits


class Model_f(nn.Module):
    def __init__(self):
        super(Model_f,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1,64,8,2,padding = 3),
            nn.ReLU(),
            nn.Conv2d(64,128,6,2),
            nn.ReLU(),
            nn.Conv2d(128,128,5,1),
            nn.ReLU(),

        )

        self.linear = nn.Sequential(
            nn.Linear(128,10),
        )

    def forward(self,x):
        output = self.conv(x)
        # output = output.view(output.shape[0],-1)
        output = torch.reshape(output,(output.shape[0],output.shape[1]*output.shape[2]*output.shape[3]))
        output = self.linear(output)
        # output is now a tensor of shape (N,10) this is pre-softmax logits of N images
        soft = nn.Softmax(dim=1)
        output = soft(output)
        return output 

    def get_logits(self,x):

        output = self.conv(x)
        # output = output.view(output.shape[0],-1)
        output = torch.reshape(output,(output.shape[0],output.shape[1]*output.shape[2]*output.shape[3]))
        output = self.linear(output)
        logits = output
        return logits  





#according to dcgan paper 
net_dim = 64
latent_dim = 100 # this is the input z dimension to the generator
ngf = 32
nc = 1 #no of channels

##### important i always forget putting commas in sequence block will output invalid syntax

class MnistGenerator(nn.Module):

    def __init__(self):
        super(MnistGenerator,self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
    
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
    
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
    
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self,x):
        output = self.block(x)
        return output
         
#output = (batch_size,1,28,28)


class MnistDiscriminator(nn.Module):
    def __init__(self, nc = 1, ndf = 32):
        super(MnistDiscriminator, self).__init__()
        self.network = nn.Sequential(
                
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
    def forward(self, input):
        output = self.network(input)
        #output = (batch_size,1)
        output = output.view(-1,1)
        # return output.view(-1, 1).squeeze(1)
        return output 
        #squeezing removes all of size 1 



        


         








