import pandas as pd
import torch

from matplotlib import pyplot as plt

from torch.utils.data import Dataset

class TestMNISTdataset(Dataset):
    def __init__(self, test_data_path):
        self.test_data = pd.read_csv(test_data_path)
        
    
    def __len__(self):
        return len(self.test_data)
    
    
    def __getitem__(self, index):
        sample_image_data = self.test_data.iloc[index][1:785]
        sample_label_data = self.test_data.iloc[index].label
        # converting to tensor and reshape
        sample_image_data = torch.FloatTensor(sample_image_data) / 255.0
        sample_image_data = torch.reshape(sample_image_data, (-1, 28, 28))
        return sample_image_data,sample_label_data


if __name__ == '__main__':
    
    
    test_data_path =  '/home/noise/Develop/MNIST/Dataset/mnist_test.csv'
    
    dataset = TestMNISTdataset(test_data_path)
    
    image, label = dataset[0]
    
    print(image, label)
    
    


