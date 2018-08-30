
""" System Defined Packages"""
import os
import sys
import torch
import torch.optim
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data
import torch.nn.parallel
from torchvision import models
"""User Defined Packages"""
import Util
from Var import params

best_acc = 0
class FineTune(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTune, self).__init__()

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

        # -- Freeze the weights ---
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FineTuneModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(FineTuneModel, self).__init__()

        self.features = original_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )

        # -- Freeze the weights ---
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        x = self.features(x)     
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    global best_acc
    # --- Loading Data for training ---
    train_loader = Util.load_train()
    val_loader = Util.load_val()

    # --- Get number of classes from train directory ---
    print(' --- Training Data Information --- ')
    classes_names = [name for name in os.listdir(params['traindir'])]
    print('Names of class: {}'.format(classes_names))
    num_classes = len(classes_names)
    print('Number of classes: {}'.format(num_classes))
    for label in classes_names:
        print('Number of examples of Class {}: {}'.format(label,
                                                          len(os.listdir(os.path.join(params['traindir'], label)))))

    # --- Get number of classes from val directory ---
    print(' --- Validation Data Information --- ')
    classes_names = [name for name in os.listdir(params['valdir'])]
    print('Names of class: {}'.format(classes_names))
    num_classes = len(classes_names)
    print('Number of classes: {}'.format(num_classes))
    for label in classes_names:
        print('Number of examples of Class {}: {}'.format(label,
                                                          len(os.listdir(os.path.join(params['valdir'], label)))))

    # --- create model ---
    print("Loading Network")
    #orignal_model = models.__dict__['vgg16'](pretrained=True)
    orignal_model = models.__dict__['resnet18'](pretrained=True)
    print(orignal_model)

    print("Modifing Network for finetune")
    model = FineTune(orignal_model, num_classes)
    print(model)


    # --- Define loss function (criterion) and optimizer""" ---

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                params['learningRate'])

    '''
    # --- Resuming traing from saved checkpoint ---
    if params['resume']:
             model, optimizer = Util.checkpoint(model)

    # --- If you want to evaluate the model ---
    if params['evaluate']:
        print("Entering the evaluation phase")
        model = Util.bestmodel(model)
        Util.validate(val_loader, model, criterion)
        return
    '''

    # --- Starting the training Process ---
    print("Entering the training phase")
    for epoch in range(params['startEpoch'], params['epochs']):

        #Util.adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        Util.train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc = Util.validate(val_loader, model, criterion)
        if acc > best_acc:
            is_best = True
            best_acc = acc
        else:
            is_best = False

        # --- Saving the model ---
        Util.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        if is_best:
            state_dict = models.state_dict()
            torch.save(state_dict, 'Models/best_model-' + str(best_acc) +  '.pth')
        else:
            pass
            # torch.save(model, 'Models/model-' + str(epoch) +  '.pth')

if __name__ == '__main__':
    main()



