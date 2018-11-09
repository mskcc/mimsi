'''
MiMSI Model

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

'''

import torch
import torch.nn as nn


class MSIModel(nn.Module):
    def __init__(self):
        super(MSIModel, self).__init__()
        self.num_features = 512
        self.batch_size = 1

        #Input is N x (3 x 100 x 40)
	    self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #32 x 100 x 40
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), #32 x 50 x 20
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), #32 x 50 x 20
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), #32 x 50 x 20
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), #32 x 50 x 20
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), #32 x 50 x 20
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2) #64 x 25 x 10
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), #64 x 25 x 10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64 x 25 x 10
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), #64 x 25 x 10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1) #64 x 25 x 10
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), #64 x 25 x 10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64 x 25 x 10
        )

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), #64 x 25 x 10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1) #64 x 25 x 10
        )

        ''' 
            This is the section of the model that takes all of the instance-level 
            feature embeddings and creates the final N x num_features vectors for 
            each instance. These instance vectors will be averaged to get the 
            sample level embedding before input into the final classification
            layer
        '''
        self.final_instance_features = nn.Sequential(
            nn.Linear(64 * 25 * 10, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
        )


        self.classifier = nn.Sequential(
            nn.Linear(self.num_features*self.batch_size, 1),
            nn.Sigmoid()
        )

	    self.relu = nn.ReLU()


    def forward(self, x):
        x = x.squeeze(0)
        # ResNet style block 1
        out_1 = self.conv1(x)
        
        # ResNet Style block 2
        res_1 = out_1
        out_2_1 = self.conv2_1(out_1)
        out_2_1 = out_2_1 + res_1
        out_2_1 = self.relu(out_2_1)
        
        res_2 = out_2_1
        out_2_2 = self.conv2_2(out_2_1)
        out_2_2 = out_2_2 + res_2
        out_2_2 = self.relu(out_2_2)

        # We only downsample once (beside initial max 
        # pooling in con1) during our modified resnet
        # because the microsatellite regions are so small
        res_3 = self.downsample(out_2_2)

        # ResNet style block 3
        out_3_1 = self.conv3_1(out_2_2)
        out_3_1 = out_3_1 + res_3
        out_3_1 = self.relu(out_3_1)

        res4 = out_3_1
        out_3_2 = self.conv3_2(out_3_1)
        out_3_2 = out_3_2 + res4
        out_3_2 = self.relu(out_3_2)

        
        # ResNet style block 4
        res_5 = out_3_2
        out_4_1 = self.conv4_1(out_3_2)
        out_4_1 = out_4_1 + res_5
        out_4_1 = self.relu(out_4_1)

        res6 = out_4_1
        out_4_2 = self.conv4_2(out_4_1)
        out_4_2 = out_4_2 + res6
        final_instance_embed = self.relu(out_4_2)

        final_instance_embed = final_instance_embed.view(-1, 64 * 25 * 10)
        I = self.final_instance_features(final_instance_embed)  # N x num_features     

        # S is the sample-level embedding, which is the aggregation (via mean) of 
        # each microsatellite instance vector
	    S = torch.mean(I, 0) 

        Y_prob = self.classifier(S)

        # We do a simple threshold at .5 to get our final label
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat



    # Helper functions
    def calculate_classification_error(self, Y_hat, Y):
        Y = Y.float()
        # convert -1 negative label to 0 for binary cross entropy calc
        if Y == -1:
            Y = 0.
        
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error


    def calculate_objective(self, X, Y):
        Y = Y.float()
        # convert -1 negative label to 0 for binary cross entropy calc
        if Y == -1:
            Y = 0.

        Y_prob, Y_hat = self.forward(X)
        
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        
        # Loss function is binary cross entropy aka neg log likelihood
        loss = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  
        
        return loss, Y_prob, Y_hat 

    

