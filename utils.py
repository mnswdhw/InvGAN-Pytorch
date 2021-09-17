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
from models import MnistCnn, Model_b, Model_f
import pickle 
from cleverhans.torch.utils import optimize_linear
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import torch.nn.functional as F
from pyt_models import Discriminator, Generator, Encoder


device = "cuda" if torch.cuda.is_available() else "cpu"
stddev = np.sqrt(1.0 / 128)
batch_size = 50
latent_dim = 128
# rec_rr = 10
rec_rr  = 10 #for invgan
init_lr = 10
# init_lr = 0.01 #for invgan recons_batch_repo
rec_iter = 200
display_steps = 20






def exp_lr_scheduler(optimizer, global_step, init_lr, decay_steps, decay_rate, lr_clip=0.1, staircase=True):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if staircase:
        lr = init_lr * decay_rate**(global_step // decay_steps)
    else:
        lr = init_lr * decay_rate**(global_step / decay_steps)
    # lr = max(lr, lr_clip)


    if global_step % decay_steps == 0:
        print(decay_steps)
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_gen():
  gen = Generator()
  gen.load_state_dict(torch.load("./pyt_models_weights/generator_200000.pth"))
  gen = gen.to(device)
  gen = gen.eval()
  return gen  #returns loaded gen in eval mode 

def load_dis():
  dis = Discriminator()
  dis.load_state_dict(torch.load("./pyt_models_weights/discriminator_200000.pth"))
  dis = dis.to(device)
  dis = dis.eval()
  return dis


def load_Cnn():
  model = MnistCnn()
  # model = Model_f()
  # model = Model_b()
  model.load_state_dict(torch.load("./checkpoints/mnist_classifier_a_10.pth"))
  model = model.to(device)  
  model = model.eval()
  return model #return loaded model in eval mode on device


def load_enc():
    enc = Encoder()
    enc.load_state_dict(torch.load("./pyt_models_weights/enc_20000.pth"))
    enc = enc.to(device)
    enc = enc.eval()
    return enc #returns loaded enc in eval mode 





class Adversarial_Dataset(Dataset):
    
    def __init__(self,transform = None):
      self.transform = transform
      image_name = "./adv/mnist/FGSM_adv_images.pickle"
      label_name = "./adv/mnist/FGSM_adv_label.pickle"

      with open (image_name, 'rb') as fp:
          self.images = pickle.load(fp)
          
      with open (label_name, 'rb') as fp:
          self.labels = pickle.load(fp)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        image = self.images[index].float()
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, label)


def load_adv_dataset():
  sample = Adversarial_Dataset()
  adversarial_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5)),
  ])
  sample = Adversarial_Dataset(transform = adversarial_transform)

  test_loader = DataLoader(
      sample,
      batch_size=batch_size,
      # num_workers=4
  )

  return test_loader



def load_clean_test(batch_size=50):

  transformation = transforms.Compose([
  #   transforms.RandomCrop(28),
  #   transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  #   transforms.Normalize((0.5), (0.5)),
  ])

  mnist = Datasets("mnist",transformation,batch_size,10000) #(dataset_name,transforms_obj, batch_size, valid_set_size)
  test_loader = mnist.load_test()

  return test_loader
        





def eval_batch(id,z,images,labels):

  model = load_Cnn()
  with torch.no_grad():
      outputs = model(images)
      _, preds = torch.max(outputs, 1)
      print(preds)
      print(labels)
      running_corrects = torch.sum(preds == labels.data)

  print(f"batch : {id} accuracy : {running_corrects/batch_size}")
  return running_corrects


#positional (without =) should be before non positional
def return_fgsm_batch_adv(x,model_fn=None,eps = 0.3,norm = np.inf,clip_min=0,clip_max=1.,y=None,targeted=False,sanity_checks=True):
  if model_fn == None:
    model_fn = load_Cnn()
    model_fn = model_fn.get_logits #returns a function that computes logits on input
  # it expects model_fn to return logits rather than softmax outputs
  x_fgm = fast_gradient_method(model_fn,x,eps,norm,clip_min,clip_max,y,targeted,sanity_checks)
  return x_fgm







