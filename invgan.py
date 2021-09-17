# from pyt_models import Generator, Encoder
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
import time 


writer = SummaryWriter(f'invgan_logs/inv+def/{rec_iter}/v20')
# writer = SummaryWriter(f'invgan_logs/inv/v2')


print(rec_iter)


def reconstruct_batch_own(batch_idx, gen, inputs,z_hat):

  #z_init = (N,128)
  z_hat = z_hat.to(device)
  z_hat.requires_grad = True
  optimizer = optim.SGD([z_hat], lr=init_lr, momentum=0.7)
  print(inputs.shape) #(N,1,28,28)
  print(rec_iter)
  for i in range(rec_iter):
      
      optimizer.zero_grad()
      z_hats_recs = gen(z_hat) #(N,1,28,28)
      #calculate loss
      diff = z_hats_recs - inputs
      image_rec_loss = diff.square()
      image_rec_loss = image_rec_loss.mean(dim = (1,2,3))
      rec_loss = image_rec_loss.sum()
      rec_loss.backward()
      optimizer.step()
      cur_lr = exp_lr_scheduler(optimizer, i, init_lr, 160, 0.1, staircase=True)

  # print(z_hats_recs[0])

  recon_images = z_hats_recs
  final_z = z_hat

  return (final_z,recon_images)


def reconstruct_batch_repo(batch_idx, gen, inputs, z_hat):
  #lr = 0.01
  # rec_rr = 1


  z_hat = z_hat.to(device)
  # z_hat.requires_grad = True
  modifier = torch.zeros((50,128))
  # print(modifier.requires_grad,modifier.is_leaf)
  modifier = modifier.to(device) #this is also an op and after this op modifier probably has requires grad set to false
  # print(modifier)
  modifier.requires_grad = True
  optimizer = optim.Adam([modifier], lr=init_lr)
  inputs = inputs.repeat_interleave(rec_rr,dim = 0) #(10*N,1,28,28)

  for i in range(rec_iter):
    optimizer.zero_grad()
    z_hats_recs = gen(z_hat + modifier) #(10*N,1,28,28)
    #calculate loss
    diff = z_hats_recs - inputs
    image_rec_loss = diff.square()
    image_rec_loss = image_rec_loss.mean(dim = (1,2,3))
    rec_loss = image_rec_loss.sum()
    rec_loss.backward()
    optimizer.step()

  recon_images = torch.stack([z_hats_recs[i * rec_rr + image_rec_loss[i * rec_rr:(i + 1) * rec_rr].argmin().item()] for i in range(batch_size)], dim=0)
  z_star = torch.stack([z_hat[i * rec_rr + image_rec_loss[i * rec_rr:(i + 1) * rec_rr].argmin().item()] for i in range(batch_size)], dim=0)

  return (z_star,recon_images)




def invgan(is_clean = True,iscombined = True,recon_batch_fn = reconstruct_batch_own):
  loader = load_clean_test()

  if is_clean:
      name = "clean"
  else:
      name = "adv"

  
  gen = load_gen()
  enc = load_enc()
  running_correct_total = 0
  epoch_size = 0


  if not iscombined:
    for batch_idx,(inputs,labels) in enumerate(loader):
      inputs = inputs.to(device)
      labels = labels.to(device)

      if not is_clean:
        print("adv")
        inputs = return_fgsm_batch_adv(inputs)
        inputs = inputs.clone().detach()
        inputs = inputs.to(device) #necessary or not since already on device ?
        img_grid = torchvision.utils.make_grid(inputs[:15])
        writer.add_image(f"xyz :", img_grid,batch_idx)
      z_init = enc(inputs)[0]
      recon_imgs = gen(z_init)
      img_grid = torchvision.utils.make_grid(inputs[:15])
      writer.add_image("inital clean input images :", img_grid,1)
      img_grid = torchvision.utils.make_grid(recon_imgs[:15])
      writer.add_image("recon  images :", img_grid,1)
      running_correct_total = running_correct_total + eval_batch(batch_idx, z_init, recon_imgs,labels)
      epoch_size = epoch_size + batch_size
      break
      
    print(f"Accuracy using invgan defense only recon_imgs: {running_correct_total/epoch_size}")

  else:

    for batch_idx,(inputs,labels) in enumerate(loader):
      inputs = inputs.to(device)
      labels = labels.to(device)
      if not is_clean:
        print("adv")
        inputs = return_fgsm_batch_adv(inputs)
        inputs = inputs.clone().detach()
        inputs = inputs.to(device) #necessary or not since already on device ?
        img_grid = torchvision.utils.make_grid(inputs[:15])
        writer.add_image(f"xyz :", img_grid,batch_idx)
      z_init = enc(inputs)[0]
      z_init_new = z_init.clone().detach()
      (z_final,recon_imgs) = recon_batch_fn(batch_idx,gen,inputs,z_init_new)
      img_grid = torchvision.utils.make_grid(inputs[:15])
      writer.add_image(f"inital {name} input images :", img_grid,batch_idx)
      img_grid = torchvision.utils.make_grid(recon_imgs[:15])
      writer.add_image(f"reconstructions of {name} input images :", img_grid,batch_idx)
      writer.close()
      running_correct_total = running_correct_total + eval_batch(batch_idx, z_final, recon_imgs,labels)
      epoch_size = epoch_size + batch_size
      break

    print(f"Accuracy using invgan+defense recon_imgs: {running_correct_total/epoch_size}")

  







if __name__ == "__main__":

  start_time = time.time()
  invgan(is_clean=True,iscombined = True,recon_batch_fn = reconstruct_batch_repo)
  end_time = time.time()

  print(end_time - start_time)




