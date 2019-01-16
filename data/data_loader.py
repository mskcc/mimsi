'''
MiMSI Data Loader

Torch Data Loader for MiMSI

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

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

    def __parsefilename__(self, filename):
        '''format of the vector filename is:
            {sample_id}_{label}_{type_of_file}.npy
            where the label is either -1 or 1 and the type of file is either "data" or "locations"

            This helper method parses out the sample_id and label, and takes into account 
            corner cases like the client using "_" in their sample name.
        '''

        split_filename = filename.split("_")
        label = split_filename[-2] #-1 index is data/locations, -2 is label
        sample_portion = split_filename[:-2]
        if len(sample_portion) > 1:
            # underscore must have been used since total length is over 3
            # need to join the sample_id back together
            sample_id = "_".join(sample_portion)
        else:
            # underscore wasn't used, just grab the first (and only) element
            sample_id = sample_portion[0]

        return sample_id, label



    def __getitem__(self, idx):
        data_file = os.path.join(self.root_dir, self.data_file_list[idx])
        loc_file = os.path.join(self.root_dir, self.data_file_list[idx]).replace("data", "locations")
        
        sample_id, label = self.__parsefilename__(self.data_file_list[idx])

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

        return bag, label, 0, sample_id


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
    
