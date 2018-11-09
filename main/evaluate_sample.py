from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics

from data.data_loader import MSIBags
from model.mi_msi_model import MSIModel

parser = argparse.ArgumentParser(description='MiMSI Sample(s) Evalution Utility')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA for use off GPU, if this is not specified the utility will check availability of torch.cuda')
parser.add_argument('--saved-model', default="mi_msi_1.model", help='name of the saved model weights to load')
parser.add_argument('--vector-location', default="./eval", help='location of generated vectors to evaluate')
parser.add_argument('--save', default=True, help='save the results of the evaluation to a numpy array')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    print('\nGPU is Enabled!')

print('Evaluating Samples, Lets go!!!')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

eval_loader = data_utils.DataLoader(MSIBags(args.vector_location, False, False),
                                     batch_size=1,
                                     shuffle=False,
                                     **loader_kwargs)
model = MSIModel()
if args.cuda:
    model.cuda()


def evaluate():
    model.eval()
    result_list = []

    with torch.no_grad():
        for batch_idx, (data, label, sample_id) in enumerate(eval_loader):
            # Since we're evaluating here we're just using a default label
            # of -1 and ignoring the loss
            bag_label = label

            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)

            # Evaluate the sample
            _, Y_prob, Y_hat = model.calculate_objective(data, bag_label)

            # Record the result as a probability 
            Y_prob = Y_prob.item()
            result = [patient, Y_prob]
            result_list.append(result)
            print(patient + "\t" + str(Y_prob) + "\n")

    if args.save:
        np.save('./' + args.name + '_results.npy', result_list)



if __name__ == "__main__":

    model.load_state_dict(torch.load(args.saved_model))
    evaluate()