import dc_gan
import torch 
from torch.utils.tensorboard import SummaryWriter
import torchvision
from pyt_models import Generator
from utils import *
# import matplotlib.pyplot as plt
# from PIL import Image 


# writer = SummaryWriter('defense_logs/gan_generated_imgs/epoch_500/v1')
writer = SummaryWriter('invgan_logs/gan_generated_imgs/v2')
# z = torch.randn(20,100,1,1)
z = torch.randn(20,128)
# gen = dc_gan.generator1
gen = Generator()
gen.load_state_dict(torch.load("./pyt_models_weights/generator_200000.pth"))
gen = gen.eval()
output = gen(z)
img_grid = torchvision.utils.make_grid(output)
writer.add_image('20_gan_generated_img', img_grid)
writer.close() ## very important since the writer is asynchronous so it needs time to write the images to the 
#checkpoint but the program ends before that hence we must wait for the writer to finish writer.close() does that




def inv_gan_vis_rec_iter_0():
  enc = load_enc()
  gen = load_gen()
  enc = enc.eval()
  gen = gen.eval()

  test_loader = load_clean_test()

  print(device)

  for (input,label) in test_loader:
    input = input.to(device)
    img_grid = torchvision.utils.make_grid(input[:15])
    writer.add_image("inital clean input images :", img_grid,1)
    output = enc(input)[0] #(N,128)
    gen_output  = gen(output)
    img_grid = torchvision.utils.make_grid(gen_output[:15])
    writer.add_image("reconstructed clean images :", img_grid,1)
    writer.close()
    break

  return







# class Defense_clean():

#   def __init__(self):
#     self.test_loader = load_clean_test()
#     self.gen = load_gen()
#     self.enc = load_enc()
#     self.model = load_Cnn()


#   def batch_accuracy(self,inputs,labels):
#     with torch.no_grad():
#       inputs = inputs.to(device)
#       labels = labels.to(device)

#       outputs = self.model(inputs)

#       _, preds = torch.max(outputs, 1)
#       running_corrects = torch.sum(preds == labels.data)

#     return running_corrects

#   def defense_on_clean_img_rec_lr_0(self):

#     running_corrects = 0.0
#     epoch_size = 0.0
#     for batch,(inputs,labels) in tqdm(enumerate(self.test_loader)):
#       inputs = inputs.to(device)
#       labels = labels.to(device)
#       latent_space = self.enc(inputs)[0]   # inputs = (N,1,28,28), latent_space = (N,128)
#       recon_imgs = self.gen(latent_space) #(N,1,28,28)
#       running_corrects = running_corrects + self.batch_accuracy(recon_imgs,labels)
#       epoch_size = epoch_size + inputs.size(0)

#     accuracy = running_corrects.double() / epoch_size

#     return accuracy 

#   def defense_on_clean_img_rec_lr_1000(self):

#     #obtain initial z 
#     # z-> G -> G(z) -> 

#     # inside the batch, z_initial = enc(inputs)
#     return 
    




# if __name__ == "__main__":

#     inv_gan_vis_rec_iter_0()

