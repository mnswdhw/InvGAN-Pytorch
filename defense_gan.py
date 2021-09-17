from pyt_models import Generator, Encoder
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import math
import os
from dataset import Datasets
from tqdm import tqdm
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
from models import MnistCnn
import pickle 
from utils import *


stddev = np.sqrt(1.0 / 128)

device = "cuda" if torch.cuda.is_available() else "cpu"
# writer = SummaryWriter('invgan_logs/defgan_only/clean/r_10_t_200_v3') #set the path
writer = SummaryWriter('defense_gan_logs/adv_def/v9')






def reconstruct_batch(batch_idx, gen, inputs):
    #inputs = (N,1,28,28)

    
    z_hat = torch.empty((batch_size*rec_rr,latent_dim)).normal_(std=stddev)  
    z_hat = z_hat.to(device)
    z_hat.requires_grad = True #so that autograd tracks it
    #defining optimizer
    optimizer = optim.SGD([z_hat], lr=init_lr, momentum=0.7)
    inputs = inputs.repeat_interleave(rec_rr,dim = 0) #(10*N,1,28,28)
    # print(inputs.shape)
    for i in range(rec_iter):
        
        optimizer.zero_grad()
        z_hats_recs = gen(z_hat) #(10*N,1,28,28)
        #calculate loss
        diff = z_hats_recs - inputs
        image_rec_loss = diff.square()
        image_rec_loss = image_rec_loss.mean(dim = (1,2,3))
        rec_loss = image_rec_loss.sum()
        rec_loss.backward()
        optimizer.step()
        cur_lr = exp_lr_scheduler(optimizer, i, init_lr, 160, 0.1, lr_clip = 0.1, staircase=True)
        
    recon_images = torch.stack([z_hats_recs[i * rec_rr + image_rec_loss[i * rec_rr:(i + 1) * rec_rr].argmin().item()] for i in range(batch_size)], dim=0)
    # img_grid = torchvision.utils.make_grid(z_hats_recs[:50])
    # writer.add_image(f"all 10 reconstructions of the first five images of the first batch:", img_grid,batch_idx)
    #shape of recon_images = (N,1,28,28)
    z_star = torch.stack([z_hat[i * rec_rr + image_rec_loss[i * rec_rr:(i + 1) * rec_rr].argmin().item()] for i in range(batch_size)], dim=0)
    
    return (z_star,recon_images)



def defense_gan(is_clean = True):
    loader = load_clean_test()
    
    if is_clean:  
        name = "clean"
    else:
        name = "adv"
     
    gen = load_gen()
    running_correct_total = 0
    epoch_size = 0

    for batch_idx,(inputs,labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(inputs.requires_grad)
        if not is_clean:
            print("adv")
            inputs = return_fgsm_batch_adv(inputs)
            inputs = inputs.clone().detach()
            inputs = inputs.to(device) #necessary or not since already on device ?
            img_grid = torchvision.utils.make_grid(inputs[:15])
            writer.add_image(f"xyz :", img_grid,batch_idx)
            
        
        (z_best,recon_imgs) = reconstruct_batch(batch_idx,gen,inputs)
        img_grid = torchvision.utils.make_grid(inputs[:15])
        writer.add_image(f"inital {name} input images :", img_grid,batch_idx)
        img_grid = torchvision.utils.make_grid(recon_imgs[:15])
        writer.add_image(f"reconstructions of {name} input images :", img_grid,batch_idx)
        writer.close()
        running_correct_total = running_correct_total + eval_batch(batch_idx, z_best, recon_imgs,labels)
        epoch_size = epoch_size + batch_size
        break

    print(f"Accuracy using recon_imgs: {running_correct_total/epoch_size}")
  

    
    

if __name__ == "__main__":

    # defense(is_clean=True)
    defense_gan(is_clean = True)
    defense_gan(is_clean = False)

   


