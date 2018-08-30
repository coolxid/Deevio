import os
import time
import shutil
import torch
import torch.optim
from tqdm import tqdm
from PIL import Image
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms # Torch package image processing 

from Var import params

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_train():
   
   dataset = datasets.ImageFolder(params['traindir'], transforms.Compose([
       transforms.Resize((224, 224), Image.ANTIALIAS),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
   
   train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batchSize'], shuffle=True,
                                              num_workers=params['workers'])
   
   return train_loader

def load_val():
   
   dataset = datasets.ImageFolder(params['valdir'], transforms.Compose([
       transforms.Resize((224, 224), Image.ANTIALIAS),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
   
   val_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batchSize'], shuffle=True,
                                            num_workers=params['workers'])

   return val_loader

def checkpoint(model, optimizer):
    if params['resume']:
        if os.path.isfile(params['resume']):
            print ("=> loading checkpoint '{}'".format(params['resume']))
            checkpoint = torch.load(params['resume'])
            params['startEpoch'] = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(params['evaluate'], checkpoint['epoch']))
        else:
            print ("=> no checkpoint found at '{}'".format(params['resume']))
    return model, optimizer

def save_checkpoint(state, is_best, filename='Models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'Models/model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = params["learningRate"] * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch):

    # switch to train mode
    model.train()
    tloader = tqdm(train_loader, total=len(train_loader))
    for i, (input, target) in enumerate(tloader):
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # --- Compute output ---
        output = model(input_var)
        loss = criterion(output, target_var)

        # --- Compute gradient and do SGD step ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print('Epoch: [{}][{}/{}]\t'
              'Loss: {}'.format(
               epoch, i, len(train_loader), loss.data[0]))


def validate(val_loader, model, criterion):
    correct = 0
    total = 0
    # --- Switch to evaluate mode ---
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # --- Compute output ---
        output = model(input_var)
        loss = criterion(output, target_var)

        # --- Measure accuracy and record loss ---
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()

        print('Test: [{}/{}]\t'
              'Loss :{}'.format(
               i, len(val_loader), loss.data[0]))

    print('Accuracy of the network on validation set: %d %%' % (
            100 * correct / total))
    return 100 * (correct / total)
