import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from train_dataset import TrainMNISTdataset
from test_dataset import TestMNISTdataset
from cnn import CNNNetwork

import tqdm



def create_train_dataloader(train_dataset, batch_size):
    train_dataloader = DataLoader(train_dataset, batch_size)
    return train_dataloader


def create_test_dataloader(test_dataset, batch_size):
    test_dataloader = DataLoader(test_dataset, batch_size)
    return test_dataloader


def train_single_epoch(model, train_dataloader, loss_fn, optimiser):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    iter = 0
    for input, target in train_dataloader:
        
        iter =+ 1
        print(f'single_epoch: {iter}')
        
        # reset gradient to zero
        optimiser.zero_grad()

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item() * input.size(0)
        _, predicted = torch.max(prediction, 1)
        total_correct += (predicted == target).sum().item()
    epoch_loss = total_loss / len(train_dataloader.dataset)
    epoch_acc = total_correct / len(train_dataloader.dataset)
    print(f'train loss: {epoch_loss} acc: {epoch_acc}')
    return epoch_loss, epoch_acc
    
    
def test_single_epoch(model, test_dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    with torch.no_grad():
        for input, target in test_dataloader:
            
            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)
            
            total_loss += loss.item() * input.size(0)
            _, predicted = torch.max(prediction, 1)
            total_correct += (predicted == target).sum().item()
        epoch_loss = total_loss / len(test_dataloader.dataset)
        epoch_acc = total_correct / len(test_dataloader.dataset)
        print(f'test loss: {epoch_loss} acc: {epoch_acc}')
        return epoch_loss, epoch_acc



def train(model, train_dataloader, test_dataloader, loss_fn, optimiser, epochs):
    for i in range(epochs):
        print(f'Epochs: {i+1}')
        train_single_epoch(model, train_dataloader, loss_fn, optimiser)
        # print(f"Epoch {epoch+1} training loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
        print('==================================')
        test_single_epoch(model, test_dataloader, loss_fn)
        # print(f"Epoch {epoch+1} testing loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")
        print('==================================')
    print('Finished training')



if __name__ == '__main__':
    
    BATCH_SIZE = 10000
    EPOCHS = 10
    LEARNING_RATE = 0.001
    train_data_path = '/home/noise/Develop/MNIST/Dataset/mnist_train.csv'
    test_data_path =  '/home/noise/Develop/MNIST/Dataset/mnist_test.csv'

    
    train_dataset = TrainMNISTdataset(train_data_path)
    test_dataset = TestMNISTdataset(test_data_path)
    
    train_dataloader = create_train_dataloader(train_dataset, BATCH_SIZE)
    test_dataloader = create_test_dataloader(test_dataset,BATCH_SIZE)
    
    model = CNNNetwork()
    
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)
    
    train(model, train_dataloader, test_dataloader, loss_fn, optimiser, EPOCHS)
    