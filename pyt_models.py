import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



def l2normalise(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)



def spectral_norm(w,u,num_iters = 1):

    w_shape = w.shape
    w = torch.reshape(w,(-1,w_shape[-1]))
    u_hat = u
    v_hat = None
    w_trans = torch.transpose(w,0,1)
    for _ in range(num_iters):
        v_ = torch.matmul(u_hat,w_trans)
        v_hat = l2normalise(v_) #(1,k*k*in_channels)

        u_ = torch.matmul(v_hat,w) 
        u_hat = l2normalise(u_) #(1,out_channels)
        u_hat_trans = torch.transpose(u_hat,0,1) #(out,1)
    
    sigma = torch.squeeze(torch.matmul(torch.matmul(v_hat,w),u_hat_trans)) #(scalar) = [1] dimension
    w_norm = w/sigma

    w_norm = torch.reshape(w_norm,w_shape) #(k,k,in,out)
    return w_norm





class Myconv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True,num_iters = 1):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.u = nn.Parameter(torch.zeros((1,out_channels)),requires_grad = False)


    def forward(self, x):

        w = self.weight.permute(2,3,1,0) #(k,k,in,out)
        w_norm = spectral_norm(w,self.u)
        w_final = w_norm.permute(3,2,0,1)
        self.weight = nn.Parameter(w_final)

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class Mylinear(nn.Linear):
    def __init__(self,in_features, out_features, bias=True):
        nn.Linear.__init__(self,in_features, out_features, bias)
        self.u = nn.Parameter(torch.zeros((1,out_features)),requires_grad = False)


    def forward(self,x):

        w = self.weight.permute(1,0) #(in,out)
        w_norm = spectral_norm(w,self.u)
        w_final = w_norm.permute(1,0) #(out,in)
        self.weight = nn.Parameter(w_final)
        return F.linear(x,self.weight,self.bias)




class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        net_dim = 64
        use_sn = False
        update_collection = None
        self.bn_linear = nn.BatchNorm1d(4*4*4*net_dim)
        self.relu = nn.ReLU()
        self.deconv_0 = nn.ConvTranspose2d(4*net_dim,2*net_dim,5,2, padding = 1)
        self.bn_0 = nn.BatchNorm2d(2*net_dim)
        self.deconv_1 = nn.ConvTranspose2d(2*net_dim,net_dim,5,2, padding = 1)
        self.bn_1 = nn.BatchNorm2d(net_dim)
        self.deconv_2 = nn.ConvTranspose2d(net_dim,1,5,2,padding = 1)
        self.linear = nn.Linear(128,4*4*4*net_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        net_dim = 64
        output = self.linear(x)
        output = self.bn_linear(output)
        output = self.relu(output)
        output = output.view(-1,4,4,4*net_dim) #NHWC
        output = output.permute(0,3,1,2) #NCHW
        output = self.deconv_0(output)
        output = output.permute(0,2,3,1) #NHWC
        output = output[:,:8,:8,:] #(8,8)
        output = output.permute(0,3,1,2)
        output = self.bn_0(output) 
        output = self.relu(output)
        output = output.permute(0,2,3,1) #NHWC
        output = output[:,:7,:7,:] #(7,7)
        output = output.permute(0,3,1,2) #NHWC
        output = self.deconv_1(output) #(15,15)
        output = output[:,:,:14,:14]#(14,14)
        output = self.bn_1(output)
        output = self.relu(output)
        output = self.deconv_2(output) #(29,29)
        output = output[:,:,:28,:28] #(28,28)
        output = self.sigmoid(output)
        
        return output



class Encoder(nn.Module):
    def __init__(self):
        net_dim = 64
        super(Encoder,self).__init__()
        self.conv0 = nn.Conv2d(1,net_dim,5,2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(net_dim,2*net_dim,5,2)
        self.conv2 = nn.Conv2d(2*net_dim,4*net_dim,5,2) #(N,256,4,4)
        self.linear = nn.Linear(4096,256)

    def forward(self,x):
        net_dim = 64
        x = torch.nn.functional.pad(x, (1,2,1,2), mode='constant', value=0)
        output = self.conv0(x)
        output = self.relu(output)
        output = torch.nn.functional.pad(output, (1,2,1,2), mode='constant', value=0)
        output = self.conv1(output)
        output = self.relu(output)
        output = torch.nn.functional.pad(output, (2,2,2,2), mode='constant', value=0) #since here in_height % stride !=0
        output = self.conv2(output)
        output = self.relu(output)
        output = output.permute(0,2,3,1)  
        output = torch.reshape(output,(output.shape[0],4*4*4*net_dim))
        output = self.linear(output)
        
        return output[:,:128],output[:,128:] #(N,128)





class Discriminator(nn.Module):
    def __init__(self):
        net_dim = 64
        super(Discriminator,self).__init__()
        self.conv0 = Myconv(1,net_dim,5,2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = Myconv(net_dim,2*net_dim,5,2)
        self.conv2 = Myconv(2*net_dim,4*net_dim,5,2)
        self.linear = Mylinear(4096,1)
    

    def forward(self,x):
        net_dim = 64
        x = torch.nn.functional.pad(x, (1,2,1,2), mode='constant', value=0)
        output = self.conv0(x)
        output = self.lrelu(output)
        # print(output.shape)
        output = torch.nn.functional.pad(output, (1,2,1,2), mode='constant', value=0)
        output = self.conv1(output)
        output = self.lrelu(output)
        # print(output.shape)
        output = torch.nn.functional.pad(output, (2,2,2,2), mode='constant', value=0)
        output = self.conv2(output)
        output = self.lrelu(output)
        # print(output.shape)
        output = output.permute(0,2,3,1)
        output = torch.reshape(output,(output.shape[0],4*4*4*net_dim))
        # print(output.shape)
        output = self.linear(output) #(N,1)
        # print(output.shape)
        # print(torch.squeeze(output).shape)

        return torch.squeeze(output) #(N)








## Weight of all above models are in pyt_models_weights.py 
## Gen is the pretrained mnist dataset generator
## Dis is the discriminator used to train the generator 
## Enc is the encoder to invert images to latent space of the generator

