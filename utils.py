# -*- coding: utf-8 -*-
"""
@author: binit_gajera
"""
import torch
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import copy

def get_count(df, cat):
    '''
    df is the dataframe and cat contains "positive" for abnormal and "negative" for normal
    Returns number of images in a study type XR_HAND which are of abnormal or normal
    '''
    return df[df['Path'].str.contains(cat)]['Count'].sum()
  
def np_V(x):
    '''convert numpy float to Variable tensor float'''    
    return Variable(torch.cuda.FloatTensor([x]), requires_grad=False)

def plot_data(data_loss, accs):
    train_acc = accs['train']
    valid_acc = accs['valid']
    train_loss = data_loss['train']
    valid_loss = data_loss['valid']
    epochs = range(len(train_acc))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1,)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, valid_acc)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Loss')

    plt.show()

def train_model(model, criterion, optimizer, dataloaders, scheduler, 
                dataset_sizes, num_epochs):
    since = time.time()
    data_cat = ['train', 'valid'] # data categories
    #Keeping track of the best model with highest validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    data_loss = {x:[] for x in data_cat} # for storing loss per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase=='train')
            running_loss = 0.0
            correct_preds = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end='\r')
                inputs = data['images'][0]
                labels = data['label'].type(torch.FloatTensor)
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                outputs = torch.mean(outputs)
                loss = criterion(outputs, labels, phase)
                running_loss += loss.data[0]
                # backward + optimize for training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                correct_preds += torch.sum(preds == labels.data)
            epoch_loss = running_loss.to(dtype=torch.float) / float(dataset_sizes[phase])
            epoch_acc = correct_preds.to(dtype=torch.float) / float(dataset_sizes[phase])
            data_loss[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print('{} Loss this epoch: {:.4f} Accuracy this epoch: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # Keeping the copy of model with the best validation accuracy
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    plot_data(data_loss, accs)
    # loading the best model after completing the training
    model.load_state_dict(best_model_wts)
    return model