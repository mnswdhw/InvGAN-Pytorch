from __future__ import print_function
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
from utils.config import load_config, gan_from_config
import os
import sys
from models.dataset_networks import mnist_discriminator
import torch.nn.functional as F


a = np.random.randn(2,28,28,1)

initial_pyt = torch.from_numpy(a)



def tf_dis_inference():
    # a = np.random.randn(2,28,28,1)
    # tf_initial = tf.convert_to_tensor(a, dtype=tf.float32)  
    # mnist_discriminator(tf_initial)

    test_mode = True
    ckpt_path = "experiments/cfgs/gans/mnist.yml"
    cfg = load_config(ckpt_path)
    gan = gan_from_config(cfg, test_mode)

    sess = gan.sess
    gan.initialize_uninitialized()
    gan.load_discriminator(ckpt_path="output/gans/mnist")


    gan._build()

    
    disc_output = gan.sess.run(
        [gan.disc_real]
    )
    
    return disc_output

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
        print(output.shape)
        output = torch.nn.functional.pad(output, (1,2,1,2), mode='constant', value=0)
        output = self.conv1(output)
        output = self.lrelu(output)
        print(output.shape)
        output = torch.nn.functional.pad(output, (2,2,2,2), mode='constant', value=0)
        output = self.conv2(output)
        output = self.lrelu(output)
        print(output.shape)
        output = output.permute(0,2,3,1)
        output = torch.reshape(output,(output.shape[0],4*4*4*net_dim))
        print(output.shape)
        output = self.linear(output) #(N,1)
        print(output.shape)
        print(torch.squeeze(output).shape)

        return torch.squeeze(output) #(N)


def p_vars(path):
    tf_path = os.path.abspath(path)  # Path to our TensorFlow checkpoint
    tf_vars = tf.train.list_variables(tf_path)
    return (tf_vars,tf_path)


def load_dis_pyt_weights():

    (init_vars,tf_path) = p_vars('output/gans/mnist/GAN.model-200000')

    model = Discriminator()

    tf_vars = []

    for name, shape in init_vars:
        # print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_vars.append((name, array))

    for name, array in tf_vars:

        if name[:13] == "Discriminator":
            name = name[14:].split('/')

            pointer = model

            l = name
            print(l)
            if len(l) == 3:
                continue


            if l[0] == "conv0" or l[0] == "conv1" or l[0] == "conv2":
                pointer = getattr(pointer,l[0])
                if l[1] == "biases":
                    pointer = getattr(pointer,"bias")
                elif l[1] == "w":
                    pointer = getattr(pointer,"weight")
                elif l[1] == "u":
                    pointer = getattr(pointer,"u")
            if l[0] == "linear":
                pointer = getattr(pointer,l[0])
                if l[1] == "W":
                    pointer = getattr(pointer,"weight")
                elif l[1] == "bias":
                    pointer = getattr(pointer,"bias")
                elif l[1] == "u":
                    pointer = getattr(pointer,"u")

            # print(array,array.shape)

            # print("Initialize PyTorch weight {}".format(name))
            if (l[1] == "W" or l[1] == "w") and l[0] != "linear":
                temp_tensor = torch.from_numpy(array)

                temp_tensor = temp_tensor.permute(3,2,0,1)
                pointer.data = temp_tensor 
            elif l[0] == "linear" and l[1] == "W":
                temp_tensor = torch.from_numpy(array)
                temp_tensor = temp_tensor.permute(1,0)
                pointer.data = temp_tensor 
            else:

                pointer.data = torch.from_numpy(array)

    return model #loaded model returned remember to put to eval mode before inference



if __name__ == '__main__':
    print(tf_dis_inference())
    model = load_dis_pyt_weights()
    model = model.eval()
    initial_pyt = initial_pyt.permute(0,3,1,2)
    initial_pyt = initial_pyt.to(torch.float32)
    print(model(initial_pyt))
