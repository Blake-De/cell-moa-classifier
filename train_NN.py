#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F
import os.path
import sys
from sklearn import preprocessing
import imageio.v2 as imageio
import numpy as np
import argparse, time
import fsspec
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from torchvision import models


labelmap = {
"Actin_disruptors": 0,
"Aurora_kinase_inhibitors": 1,
"Cholesterol-lowering": 2,
"DMSO": 3,
"DNA_damage": 4,
"DNA_replication": 5,
"Eg5_inhibitors": 6,
"Epithelial": 7,
"Kinase_inhibitors": 8,
"Microtubule_destabilizers": 9,
"Microtubule_stabilizers": 10,
"Protein_degradation": 11,
"Protein_synthesis": 12
}

class ImgDataset(torch.utils.data.Dataset):
    '''Dataset for reading in images from a training directory'''
    def __init__(self, args):
        '''Initialize dataset by reading in image locations'''
        with fsspec.open_files(args.train_data_dir+'/TRAIN',mode='rt')[0] as f:
            self.examples = [] # list of (label, [red,green,blue files])
            n_classes = len(labelmap)
            for line in f:
                label, c1, c2, c3 = line.rstrip().split(' ')
                # create absolute paths for image files
                self.examples.append((labelmap[label], [ args.train_data_dir + '/' + c for c in (c1,c2,c3)]))
    
        # Transfomations 
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15)
        ]) 
    
    def open_image(self,path):
        '''Return img at path, caching downloaded images'''
        fname =  path.rsplit('/',1)[-1]
        if path.startswith('gs://'): # check for downloaded file
            if os.path.exists(fname):
                path = fname
        if path.startswith('gs://'): #cache download
            with fsspec.open_files(path,mode='rb')[0] as img:
                out = open(fname,'wb')
                out.write(img.read())
                out.close()
                path = fname
        return  imageio.imread(open(path,'rb'))

    def __len__(self):
        return len(self.examples)
      
    def __getitem__(self, idx):
        imgs = [self.open_image(fname) for fname in self.examples[idx][1]]
        # Consider applying an image transform
        imgs = np.array(imgs, np.float32).transpose(1, 2, 0)
        imgs = self.transform(imgs)
        
        return {
          'img': imgs,
          'label': self.examples[idx][0]}       


# Define my network - these are not necessarily reasonable hyperparameters
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Define model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Use defualt initial weights
        
        # Get the final feature map or neruons and plug it into the the final layer.
        final_features = self.resnet.fc.in_features
        self.resnet.fc =  nn.Linear(final_features, 13) # this already flattens it.
           
    
    def forward(self, x):
        # Lazy normalization
        x = x / 255.0
        # Apply model
        out = self.resnet(x)
    
        #For evaluation we want a softmax - you must return this as
        #the first element of a tuple.  For training, in order to use
        #the numerically more stable cross_entropy loss we will also return
        #the un-softmaxed values
        return F.softmax(out,dim=1),out

    
def weight_sampler(dataset):
    labels = [example[0] for example in dataset.examples]
    counts = np.bincount(labels)
    
    # Increase the weight of rare classes
    weights = 1.0 / counts
    sample_weights = [weights[l] for l in labels]
    
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    
# Output model/train
def run_training(args):

    # Read the training data
    dataset = ImgDataset(args)
    sampler = weight_sampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             sampler=sampler, 
                                             num_workers=8,
                                             pin_memory=True
                                            )
  
    # Create an instance of the model
    model = MyModel().to('cuda')
    loss_object = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for e in range(args.max_epochs):
        start = time.time()
        for i,batch in enumerate(dataloader):

            # Model training loop
            model.train()
            optimizer.zero_grad()

            # Get images and labels
            images = batch['img'].to('cuda')
            labels = batch['label'].to('cuda')

            # Forwards pass and loss
            softmax_out, out = model(images)
            loss = loss_object(out, labels)
            losses.append(loss.item())

            # Backwards and update model
            loss.backward()
            optimizer.step()

            # Write the summaries and print an overview fairly often.
            if i % 100 == 0: # this is too often
                # Print status to stdout.
                print('Epoch %d Step %d: loss = %f' % (e,i, losses[-1]))
                sys.stdout.flush()
        print("Epoch time:",time.time()-start)
        start = time.time()
    # Export the model so that it can be loaded and used later for predictions.
    # For maximum compatibility export a trace of an application of the model  
    testdataloader = torch.utils.data.DataLoader(dataset,batch_size=1) # one example at a time for testing
    testbatch = next(iter(testdataloader))
    with torch.no_grad():
        model.eval()
        traced = torch.jit.trace(model, testbatch['img'].to('cuda'))
  
    torch.jit.save(traced,args.out)


if __name__ == '__main__':
    # Basic model parameters as external flags.
    parser = argparse.ArgumentParser('Train a model.')
    parser.add_argument('--max_epochs', default=1, type=int, help='Maximum number of epochs to train.')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size.')
    parser.add_argument('--train_data_dir', default='gs://mscbio2066-data/trainimgs', help='Directory containing training data')
    parser.add_argument('--out', default='model.pth', help='File to save model to.')

    # Feel free to add additional flags to assist in setting hyper parameters
    args = parser.parse_args()
    run_training(args)
