'''
MiMSI Train/Test

Used to train and validate a MiMSI model from scratch

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

'''



import numpy as np
import os
import sys

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics

from ..data.data_loader import MSIBags
from ..model.mi_msi_model import MSIModel

# Training settings
parser = argparse.ArgumentParser(description='MiMSI - A Multiple Instance Learning Model for detecting microsatellite instability in NGS data')
parser.add_argument('--epochs', type=int, default=40, metavar='N', help='Number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='Learning rate used in training (default: 0.0001)')
parser.add_argument('--reg', type=float, default=5e-4, metavar='R', help='Weight decay used in training (default: 5e-4)')
parser.add_argument('--seed', type=int, default=2, metavar='S', help='Random Seed (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training for use off GPU, if this is not specified the utility will check availability of torch.cuda')
parser.add_argument('--name', default="mi_msi_1", help='Name of the model, ')
parser.add_argument('--train-location', default="./main", help='Directory Location for Training Data')
parser.add_argument('--test-location', default="./main", help='Directory Location for Testing Data')
parser.add_argument('--save', default=False, help='Save the model weights to disk after training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


print('Lets Go!!!\n')

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is enabled!')

print('Loading Training and Test Set\n')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(MSIBags(args.train_location, True, True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(MSIBags(args.test_location, True, True),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

model = MSIModel()
if args.cuda:
    model.cuda()


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label, locations, sample_id) in enumerate(train_loader):
        bag_label = label
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()

        # calculate loss and error
        loss, Y_prob, Y_hat = model.calculate_objective(data, bag_label)
        train_loss += loss.item()
        
        error = model.calculate_classification_error(Y_hat, bag_label)
        train_error += error

        # back prop
        loss.backward()

        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))
    return train_loss, train_error

def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    
    tp = 0.
    fp = 0.
    tn = 0.
    fn = 0.
    
    labels = []
    preds = []

    incorrect = []

    if len(test_loader) == 0:
        print('No testing data supplied! Please indicate a directory containing generated NGS vectors in .npy format.')
        return
        
    with torch.no_grad():
        for batch_idx, (data, label, locations, sample_id) in enumerate(test_loader):
            bag_label = label
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            
            # Do our prediction
            loss, Y_prob, Y_hat = model.calculate_objective(data, bag_label)
            test_loss += loss.item()
        
            # save prediction and labels for analysis, if required
            preds.append(Y_prob.item())
            labels.append(bag_label.item())
 
            # calculate error
            error = model.calculate_classification_error(Y_hat, bag_label)
            
            if error == 0. and Y_hat == 1.:
                tp += 1
            elif error == 0. and Y_hat == 0.:
                tn += 1
            elif error == 1. and Y_hat == 1.:
                fp += 1
                # save the incorrect cases for further analysis
                incorrect.append((patient, Y_prob.item()))
            else:
                fn += 1
                # save the incorrect cases for further analysis
                incorrect.append((patient, Y_prob.item()))

            test_error += error


    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    auroc = metrics.auc(fpr, tpr)
    if args.save:
        np.save('./' + args.name + '_fpr.npy', fpr)
        np.save('./' + args.name + '_tpr.npy', tpr)
        np.save('./' + args.name + '_thres.npy', thresholds)
    
    
    print('Incorrect cases (based on provided labels): \n')
    print(incorrect)

    print('Test Set, Loss: {:.4f}, Test error: {:.4f}\n'.format(test_loss, test_error))
    print('TP, FP, TN, FN: {:.4f},{:.4f},{:.4f},{:.4f}\n'.format(tp, fp, tn, fn))
    print('AUROC: {:.4f}\n'.format(auroc))

if __name__ == "__main__":
    train_loss_list = []
    train_error_list = []
    
    # Checkpoint for our model so that we can save the best
    # performing model during training
    best_checkpoint = {
        'epoch': 0,
        'loss': 99999,
        'error': 999999,
        'state_dict': {},
        'optimizer': {}
    }

    if len(train_loader) == 0:
        print('No training data supplied! Please indicate a directory containing generated NGS vectors in .npy format.')
        sys.exit() 

    print('Training the Model... \n')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_error = train(epoch)
        train_loss_list.append(train_loss)
        train_error_list.append(train_error)

        # If this set of weights beats our best loss and error thus far, set the checkpoint
        if train_loss < best_checkpoint['loss'] and train_error < best_checkpoint['error']:
            best_checkpoint = {
                'epoch': epoch,
                'loss': train_loss,
                'error': train_error,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

    if args.save:
        np.save('./' + args.name + '_trainerror.npy', train_error_list)
        np.save('./' + args.name + '_trainloss.npy', train_loss_list)
        train_loss_list = None
        train_error_list = None
    
    print('Training Complete... \n')

    print('Testing the Model... \n')
    model.load_state_dict(best_checkpoint['state_dict'])

    test()

    if args.save:
        model.cpu()
        torch.save(model.state_dict(), './' + args.name + '.model')

