import io
import torch
import urllib
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
from main_CNN import *

def read_test(image_path):

    fd = urllib.request.urlopen(image_path)
    image_file  = io.BytesIO(fd.read())
    image = Image.open(image_file)
    #image = Image.open(image_path).convert('RGB')

    test_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.ANTIALIAS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = test_transform(image)
    image.unsqueeze_(0)
    return image

def do_pred(url):


    X = read_test(url)
    # --- create model ---
    num_classes = 2
    print("Loading Network")
    # orignal_model = models.__dict__['vgg16'](pretrained=True)
    orignal_model = models.__dict__['resnet18'](pretrained=True)
    print(orignal_model)

    print("Modifing Network for finetune")
    model = FineTune(orignal_model, num_classes)
    print(model)

    model = torch.load('Models/best_model.pth')
    #model.load_state_dict(weights)

    model.eval()

    input_var = torch.autograd.Variable(X, volatile=True)
    output = model(input_var)
    _, predicted = torch.max(output.data, 1)

    if predicted[0] == 1:
        return 'good'
    else:
        return 'bad'


if __name__ == '__main__':
    #url = 'Data/train/bad/1522141919_bad.jpeg'
    #url = 'Data/train/good/1522072948_good.jpeg'
    do_pred(url)
    #http://127.0.0.1:5000/predict?image=http://127.0.0.1/Data1/bad/1522141919_bad.jpeg


