from __future__ import print_function
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
from utils.config import load_config, gan_from_config
import os
import sys

#below is the same initial input so that both the models take the same input to compare their outputs and debug.
a = np.random.randn(2,28,28,1)


initial = tf.constant(a)
initial_pyt = torch.from_numpy(a)

def tf_enc_inference():

    # below is used to print the shapes of the outputs of the encoder without running any session or building any graph 
    # this is because, in tf1 to view the values of the tensor we need to return them from a session in tf2 it is like pytorch 
    # due to eager execution we do not need to build a graph, nor do we need to run any session. Just by printing the tensors during 
    #pass we can see their output/values of the tensors.

    # mnist_encoder(tf_initial,is_training= False)


    test_mode = True
    config_path = "experiments/cfgs/gans_inv/mnist.yml"
    cfg = load_config(config_path)
    gan = gan_from_config(cfg, test_mode)


    sess = gan.sess
    gan.initialize_uninitialized()
    gan.load_encoder(ckpt_path="output/gans_inv_notrain/mnist")


    gan._build()

    
    output_tf_enc, output_enc_layers = gan.sess.run(
        [gan.encoder_latent_before,gan.output_enc_layers], feed_dict={gan.encoder_training: False, gan.discriminator_training: False},
    )

  
    return output_tf_enc,output_enc_layers





# encoder architecture pytorch
output_layers_pyt = {}

def p_vars(path):
    tf_path = os.path.abspath(path)  # Path to our TensorFlow checkpoint
    tf_vars = tf.train.list_variables(tf_path)
    return (tf_vars,tf_path)



class Encoder(nn.Module):
    def __init__(self):
        net_dim = 64
        super(Encoder,self).__init__()
        self.conv0 = nn.Conv2d(1,net_dim,5,2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(net_dim,2*net_dim,5,2)
        self.conv2 = nn.Conv2d(2*net_dim,4*net_dim,5,2) #(N,256,4,4)
        self.linear = nn.Linear(4096,256)
        # self.linear = nn.Linear(4096,128) # even if i define the output no of features to be 128 it will still output 256 as the dimension of the weight loaded is (4096,256) hence this initilaised value will be overridden by using the dimensions from the weights which is 256
    def forward(self,x):
        output_layers_pyt["input"] = x
        net_dim = 64
        x = torch.nn.functional.pad(x, (1,2,1,2), mode='constant', value=0)
        output = self.conv0(x)
        output_layers_pyt["conv0"] = output.permute(0,2,3,1)
        # print(output.shape)
        output = self.relu(output)
        output = torch.nn.functional.pad(output, (1,2,1,2), mode='constant', value=0)
        output = self.conv1(output)
        output_layers_pyt["conv1"] = output.permute(0,2,3,1)
        # print(output.shape)
        output = self.relu(output)
        output = torch.nn.functional.pad(output, (2,2,2,2), mode='constant', value=0) #since here in_height % stride !=0
        output = self.conv2(output)
        output_layers_pyt["conv2"] = output.permute(0,2,3,1) # to match with the tf predictions 
        # print(output.shape)
        output = self.relu(output)
        ## in tf version the input shape was (2,4,4,256) this was reshaped to (2,4096) while in pyt the input shape is (2,256,4,4) which is being reshaped to (2,4096) hence to 
        ## mimic the tf output we need to first permute the tensor before reshaping.
        output = output.permute(0,2,3,1)  
        # output = output.view(-1,4*4*4*net_dim)
        output = torch.reshape(output,(2,4*4*4*net_dim))
        output_layers_pyt["after_reshape"] = output
        # print(output.shape)
        output = self.linear(output)
        # print(output.shape)
        output_layers_pyt["final"] = output
        
        return output[:,:128],output[:,128:] #(N,128)
    

        
    
def load_enc_pyt_weights():

    path = '/home/manas/Desktop/projects/sigmared/invgan/invgan/output/gans_inv_notrain/mnist/GAN.model-20000'
    init_vars,tf_path = p_vars(path)
    
    model = Encoder()

    tf_vars = []

    for name, shape in init_vars:
        # print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_vars.append((name, array))

    for name, array in tf_vars:

        if name[:7] == "Encoder":
            name = name[8:].split('/')

            pointer = model

            l = name
            # print(l)
            if len(l) == 3:
                continue


            if l[0] == "conv0" or l[0] == "conv1" or l[0] == "conv2":
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

            # print(array.shape)
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



def encoder_inference(x,model):

    model = model.eval() # not necessary as no batch_norm or dropout layers in encoder but can be done does not change things.
    (output,_) = model(x) #returns two tensors of shape (2,128) each 
    return output


def is_initial_same(pyt_initial,tf_initial):
    pyt_initial = pyt_initial.permute(0,2,3,1)
    print(pyt_initial.shape)
    a = pyt_initial.detach().numpy()

    b = tf_initial.numpy()

    return np.array_equal(a,b)




if __name__ == '__main__':

    model = load_enc_pyt_weights()
    pyt_initial = initial_pyt #(2,28,28,1)
    pyt_initial = pyt_initial.permute(0,3,1,2) #(2,1,28,28)
    pyt_initial = pyt_initial.float()
    print(pyt_initial.dtype)
    # tf_initial = tf.convert_to_tensor(initial, dtype=tf.float32)

    # if is_initial_same(pyt_initial,tf_initial) == False:
    #     raise ValueError("Initial tensors are not same")
    
    output = encoder_inference(pyt_initial,model) #same initial defined global above 

    samples,b = tf_enc_inference()
    a = output_layers_pyt
    # a = output_layers_pyt
    # print(samples,samples.shape)
    # print(output,output.shape)
    # print(b["conv0"])
    # a = a["final"]
    # a = a["deconv_1"]
    # print(np.transpose(b["input"],(0,3,1,2)),np.transpose(b["input"],(0,3,1,2)).shape)
    # print(b["conv0"],b["conv0"].shape)
    # print(a["conv0"],a["conv0"].shape)
    # print(b,b.shape,type(b))
    # print(a,a.shape,type(a.detach().numpy()))
    # save_images(np.squeeze(a.detach().numpy()), "save_path")
    # print(a["after_reshape"])
    # print(b["final"].shape)

    print(a["final"])
    
    print(np.amax(np.abs( b["final"]- (a["final"]).detach().numpy())))
