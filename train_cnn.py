import torch
import torch.nn as nn
from models import MnistCnn, Model_b, Model_f
# from torchsummary import summary
from dataset import Datasets
import torchvision.transforms as transforms
import torch.optim as optim
import copy
import os
import numpy as np
import pdb


def train(model,total_epoch,optimizer,criterion,device,model_path,train_loader,val_loader):

    # pdb.set_trace()
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_acc = 0.0

    for epoch in range(total_epoch):

        model.train()  # Set model to training mode it is imp as we are also evaluating in each epoch using validation set
        
        running_loss = 0.0
        running_corrects = 0.0
        epoch_size = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):

            #moving each input,label batch tensor to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            logits = model.get_logits(inputs)  
            loss = criterion(logits, labels) #wrong as nn.Crossentropyloss expects the predictions or herer outputs to be raw unnormalised which is also called as the logits
            
            loss.backward()
            
            _, preds = torch.max(outputs, 1)

            optimizer.step()

            # print statistics
            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            
            epoch_size += inputs.size(0)
            
        # Normalizing the loss by the total number of train batches
        
        running_loss /= epoch_size
        running_corrects =  running_corrects.double() / epoch_size
        
        train_loss.append(running_loss)
        train_acc.append(running_corrects)
        
        print('train Loss: {:.4f} Acc: {:.4f}'.format(running_loss, running_corrects))
        
        # evalute
        print('Finished epoch {}, starting evaluation'.format(epoch+1))

        model.eval()   # Set model to evaluate mode
        
        running_loss = 0.0
        running_corrects = 0.0
        epoch_size = 0.0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                # print(labels)
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                logits = model.get_logits(inputs)
                # print(logits)

                loss = criterion(logits, labels)

                _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                
                epoch_size += inputs.size(0)
        
        running_loss /= epoch_size
        running_corrects =  running_corrects.double() / epoch_size
        
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(running_loss, running_corrects))
        
        val_loss.append(running_loss)
        val_acc.append(running_corrects)


    print('==> Finished Training ...')
    torch.save(model.state_dict(), model_path)




if __name__ == "__main__":
    print("hello")

    ## we can add other datasets later by adding a config functionality and loading the config file
    total_epoch = 10
    batch_size = 50 
    valid_size = 10000
    # model = MnistCnn()
    # model = Model_f()
    model = Model_b()
    

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # summary(model, input_size = (1,28,28), device = 'cpu')

    #moving the model to GPU
    model = model.to(device)
    model_folder = os.path.abspath('./checkpoints')

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model_path = os.path.join(model_folder, 'mnist_classifier_b_10.pth')

    transformation = transforms.Compose([
    # transforms.RandomCrop(28),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5)),
])

    mnist = Datasets("mnist",transformation,batch_size,valid_size)
    trainloader,val_loader = mnist.load_train()

    train(model,total_epoch,optimizer,criterion,device,model_path,trainloader,val_loader)

