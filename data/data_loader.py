'''
MiMSI Data Loader

Torch Data Loader for MiMSI

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

'''

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import normalize
from torchvision import transforms

class MSIBags(Dataset):
    def __init__(self, root_dir, include_locations=True, labeled=True):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.is_labeled = labeled
        self.include_locations = include_locations
        self.data_file_list = filter(lambda x: "data" in x, self.file_list) #parse out only samples

        self.loc_file_list = []
        if self.include_locations:
            self.loc_file_list = filter(lambda x: "locations" in x, self.file_list) #parse out location files

    def __len__(self):
        return len(self.data_file_list)


    def __getitem__(self, idx):
        data_file = os.path.join(self.root_dir, self.data_file_list[idx])

        if self.include_locations:
            loc_file = os.path.join(self.root_dir, self.data_file_list[idx])

        if self.is_labeled:
            label = int(self.data_file_list[idx].split("_")[1])
        else:
            label = -1
        
        sample_id = self.data_file_list[idx].split("_")[0]
        bag = np.load(data_file)
        bag = [np.concatenate((entry,np.zeros((100,40-entry.shape[1],3))), axis=1) for entry in bag]
        bag = np.array(bag)
        
        bag_mean = bag.mean(axis=(0, 1, 2), keepdims=True)
        bag_std = bag.std(axis=(0, 1, 2), keepdims=True)
        
        # Normalize bag
        bag = (bag - bag_mean)/(bag_std)
        
        # Setup bag into Torch Tensor-Style Dimensions
        bag = np.swapaxes(bag, 2, 3)
        bag = np.swapaxes(bag, 1, 2)
        bag = torch.from_numpy(bag).float()

        if self.include_locations:
            locations = np.load(loc_file)
            return bag, label, locations, sample_id

        return bag, label, None, sample_id


if __name__ == "__main__":
    '''
    This is a simple test you can run to verify the training
    data is successfully loaded from the filesystem using the
    DataLoader above. Prints out a success message as well as 
    the dataset mean and standard deviation upon success
    '''
    train_loader = DataLoader(MSIBags("./data", True, True),
                                     batch_size=1,
                                     shuffle=True)         
   
  
    bag_means = []
    bag_stds = []
    for batch_idx, (data, label, locations, sample) in enumerate(train_loader):
        data = data.squeeze(0)
        
        bag_mean = data.numpy().mean(axis=(0, 2, 3))
        bag_std = data.numpy().std(axis=(0, 2, 3))
        
        bag_means.append(bag_mean)
        bag_stds.append(bag_std)


    means = np.array(bag_means)
    
    stds = np.array(bag_stds)
    

    final_mean = means.mean(axis=0)
    final_std = stds.mean(axis=0)

    print("Data from training loaded successfully. \n")
    print("Final Mean: \n")
    print(final_mean)
    print("Final Standard Deviation: \n")
    print(final_std)
    
