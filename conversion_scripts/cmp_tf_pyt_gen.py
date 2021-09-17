
from __future__ import print_function
import tensorflow as tf
from utils.config import load_config, gan_from_config
from models.dataset_networks import mnist_generator
import numpy as np
from PIL import Image as im
import sys
import torch
import torch.nn as nn
import tensorflow as tf
import os 
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.misc import imsave
np.set_printoptions(threshold=sys.maxsize)


tf.compat.v1.disable_v2_behavior()



Z = [[-1.9283,  2.7688,  1.9123,  0.5134,  1.1493, -2.0589,  0.3083, -2.0969,
          1.5133,  1.3078, -0.1938, -1.6718,  1.0918, -0.6199,  0.9041, -1.0653,
          0.2895,  1.5948, -0.6112, -0.1551,  0.5589, -1.3196,  0.0529, -1.2720,
          0.1800, -2.0774, -0.9885,  0.2435,  0.8936, -1.9041, -0.0480, -0.0672,
         -0.2584, -1.6203,  1.2159,  0.9329, -1.5149,  1.9205, -0.0867, -0.4226,
         -0.5559, -0.4553, -0.2159, -0.2824,  0.0067,  1.0483, -1.8004,  0.7973,
         -0.4099,  1.3159, -1.0721,  1.1688, -0.5447, -1.1157,  1.6924,  0.6475,
         -0.1790, -0.0585, -0.5682,  1.2534,  0.3059, -0.6895, -0.5594, -0.6315,
          1.4431,  0.8037, -1.1780, -3.3634, -1.0905,  0.3235,  0.4182, -0.4271,
          0.8537,  0.0241, -0.8704,  0.2980, -1.1254,  0.7356,  0.8890,  0.9726,
         -0.2286, -1.2240, -0.7902, -0.8921,  0.2725, -0.0183, -0.7689,  0.5173,
          1.1227, -0.4582, -1.0992, -1.1382,  0.7970,  0.3993, -0.7044,  1.8757,
         -0.2555,  0.9405, -0.1748, -0.7677, -0.9033,  0.3615, -0.4695, -0.0857,
         -0.0753, -0.8706,  1.4214,  0.7415,  0.6215, -0.2851, -0.7456,  0.3459,
         -0.9828, -0.0330, -0.5048,  1.0308,  0.8042,  0.5502, -0.6126,  0.5251,
          0.3975, -0.1305, -0.4978, -0.2209, -1.1550, -0.0376, -0.6591, -0.7800],
        [ 0.7428,  2.0105, -0.9835,  0.5920,  0.7965, -1.3262,  0.4229,  0.8360,
          0.7642,  0.1448, -0.3501,  0.0717, -2.1135,  0.0059, -0.4323,  0.0787,
          0.6055,  0.0811, -1.4112,  0.0576, -0.4598,  0.3889,  0.8371,  3.4993,
         -1.5743,  0.9668,  0.0926,  0.3715,  0.8899, -0.3950, -0.1339,  2.0859,
         -2.7667, -1.1946,  0.0768,  0.8880,  0.6156,  0.3904,  0.3018, -0.3333,
         -1.0623, -1.9733,  0.3245,  1.8893, -1.0704, -0.3712, -0.8592,  1.8665,
          1.2664, -0.3463,  0.8792, -0.9145, -1.3785,  0.6663, -0.0097, -2.0127,
         -0.6964, -2.6171,  0.1033,  0.5692,  1.1947,  0.0432, -0.2141,  0.7920,
         -0.0754,  0.3429, -1.2375,  1.6103, -0.6346,  0.2630,  0.6389,  0.3970,
         -1.3343,  1.4615,  0.4728, -0.6481,  0.2730,  1.4107, -0.4004, -0.6128,
          0.7553, -1.0540,  0.5715,  0.3378,  0.6097, -0.8205,  2.1021, -1.1119,
          0.3928,  0.3493,  0.9750,  1.2226, -0.7350, -0.9588,  0.0474, -0.9106,
          1.4326,  0.6924, -0.5693,  1.0242,  0.1555, -1.4326, -0.9242, -1.5584,
         -0.4377,  3.1167, -1.0250, -0.0388,  1.1994,  0.4246, -0.9651,  0.9338,
         -1.2601,  1.0314, -0.0330, -0.0106, -2.3576, -1.5177,  0.2608,  0.0128,
         -0.2360, -0.3022,  2.4710,  1.1538, -0.3755, -0.5416, -1.3389,  0.0478]]




def save_images(X, save_path):
    print(type(X))
    print(X.dtype)
    print(X.shape)
    print("hello bitches")
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')
    print(X.shape)
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows
    print(X.ndim)
    print(nh,nw)
    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        if X.shape[1] == 3:
            X = X.transpose(0,2,3,1)

        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    print(img)
    print(img.shape)
    # imsave(save_path, img)
    data = im.fromarray(img)
    data = data.convert("L")
    data.save('match_5.png')




def tf_gen_inference():

    test_mode = True
    ckpt_path = "experiments/cfgs/gans_inv/mnist.yml"
    cfg = load_config(ckpt_path)
    gan = gan_from_config(cfg, test_mode)


    sess = gan.sess
    gan.initialize_uninitialized()
    gan.load_generator(ckpt_path="output/gans/mnist")


    gan._build()

    
    samples,output_layers = gan.sess.run(
        [gan.x_hat_sample,gan.output_layers], feed_dict={gan.encoder_training: False, gan.discriminator_training: False},
    )

    # print(type(samples))
    # print(samples.dtype)
    # print(samples.shape)
    # print(samples)
    # print("this is the new ",inspected_value.shape)

    # tf.print(inspected_value, summarize = -1, output_stream=sys.stdout)
    # print(inspected_value)
    gan.save_image(samples, 'gen1.png')
    return samples,output_layers
    # print(samples.shape)


    




##PYTORCH CODE 

output_layers_pyt = {}

def p_vars(path):
    tf_path = os.path.abspath(path)  # Path to our TensorFlow checkpoint
    tf_vars = tf.train.list_variables(tf_path)
    print(tf_vars)
    return (tf_vars,tf_path)


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
        output_layers_pyt["linear"] = output
        # print("lienar_output",output.shape)
        # print("lienar_output",output)
        output = self.bn_linear(output)
        output_layers_pyt["bn_linear"] = output
        # print("bn_linear_output",output.shape)
        # print("bn_linear_output",output)
        output = self.relu(output)
        # print("relu_output",output.shape)
        # print("relu_output",output)
#         output = output.view(-1,4*net_dim,4,4)
        output = output.view(-1,4,4,4*net_dim) #NHWC done to compare outputs

        # print(output,output.shape)
        output_layers_pyt["first_reshape"] = output
        output = output.permute(0,3,1,2) #NCHW
        # print(output)
        output = self.deconv_0(output)
        #output_shape = (9,9)
        output = output.permute(0,2,3,1) #NHWC
        output = output[:,:8,:8,:] #(8,8)
        output_layers_pyt["deconv_0"] = output
        output = output.permute(0,3,1,2)
#         a = output
        # print("deconv0_output",output.permute(0,2,3,1).shape) #print as NHWC to compare
        # print("deconv0_output",output.permute(0,2,3,1)) #print as NHWC to compare
        output = self.bn_0(output) 
        output_layers_pyt["bn_0"] = output
        output = self.relu(output)
        # print("bn_0_output",output.shape)
        # print("bn_0_output",output)
        output = output.permute(0,2,3,1) #NHWC
        output = output[:,:7,:7,:] #(7,7)
        output_layers_pyt["sliced"] = output
        output = output.permute(0,3,1,2) #NHWC
        # print(output.shape,"slicing")
        output = self.deconv_1(output) #(15,15)
        output = output[:,:,:14,:14]#(14,14)
        output_layers_pyt["deconv_1"] = output
        # print("deconv1_output",output.permute(0,2,3,1).shape)
        # print("deconv1_output",output.permute(0,2,3,1))
        output = self.bn_1(output)
        output_layers_pyt["bn_1"] = output
        # print("bn1_output",output.shape)
        # print("bn1_output",output)
        output = self.relu(output)
        # print("relu_output",output.shape)
        # print("relu_output",output)
        output = self.deconv_2(output) #(29,29)
        output = output[:,:,:28,:28] #(28,28)
        output_layers_pyt["deconv_2"] = output
        # print("decinv2_output",output.permute(0,2,3,1).shape)
        # print("deconv2_output",output.permute(0,2,3,1))
        output = self.sigmoid(output)
        # print("digmoid_output",output.permute(0,2,3,1).shape)
        # print("sigmoid_output",output.permute(0,2,3,1)) #(N,28,1,1)
        output_layers_pyt["final"] = output
        
        return output


def load_gen_weights():

    path = "/home/manas/Desktop/projects/sigmared/invgan/invgan/output/gans/mnist/GAN.model-200000"
    init_vars,tf_path = p_vars(path)

    model = Generator()

    tf_vars = []

    for name, shape in init_vars:
        # print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_vars.append((name, array))

    for name, array in tf_vars:

        if name[:9] == "Generator":
            name = name[10:].split('/')

            pointer = model

            l = name
            print(l)
            if len(l) == 3:
                continue

            if l[0] == "bn_0" or l[0] == "bn_1" or l[0] == "bn_linear":
                pointer = getattr(pointer,l[0])
                if l[1] == "beta":
                    pointer = getattr(pointer,"bias")
                elif l[1] == "gamma":
                    pointer = getattr(pointer,"weight")
                elif l[1] == "moving_mean":
                    pointer = getattr(pointer,"running_mean")
                elif l[1] == "moving_variance":
                    pointer = getattr(pointer,"running_var")
            if l[0] == "deconv_0" or l[0] == "deconv_1" or l[0] == "deconv_2":
                pointer = getattr(pointer,l[0])
                if l[1] == "biases":
                    pointer = getattr(pointer,"bias")
                elif l[1] == "w":
                    pointer = getattr(pointer,"weight")
            if l[0] == "linear":
                pointer = getattr(pointer,l[0])
                if l[1] == "W":
                    pointer = getattr(pointer,"weight")
                elif l[1] == "bias":
                    pointer = getattr(pointer,"bias")

            print(array.shape)

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

    return model # this is the loaded model 


def gen_inference(z,model):

    model = model.eval()
    output = model(z)
    return output
    








if __name__ == '__main__':

    model = load_gen_weights()
    z = torch.Tensor(Z)
    output = gen_inference(z,model)
    samples,b = tf_gen_inference()
    a = output_layers_pyt

    # print(a["first_reshape"],b["first_reshape"])
    # b = b["final"]
    # a = a["final"].permute(0,2,3,1)
    # a = a["deconv_1"]
    print(a["final"])
    a = a["final"].permute(0,2,3,1)
    b = b["final"]

    # print(b,b.shape,type(b))
    print(a,a.shape,type(a.detach().numpy()))
    # save_images(np.squeeze(a.detach().numpy()), "save_path")

    print(np.amax(np.abs( b- a.detach().numpy())))





