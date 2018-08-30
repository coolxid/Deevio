import os


#Path = os.path.relpath('./Master-Thesis/Stage-1/DataSet/imagenet')
Path = 'Data/'
traindir = Path + '/train'
valdir = Path + '/val'
Path1 = 'C:/Users/Rashid Saleem/PycharmProjects/Deevio'
directory = os.path.dirname("Models/")
if not os.path.exists(directory):
    os.makedirs(directory)
resumedir = directory + '/checkpoint.pth.tar'
bestmodeldir = directory + '/model_best.pth.tar'
params = {'workers': 2, 'epochs': 90, 'startEpoch': 0, 'batchSize': 20, 'learningRate': 0.00001, 'momentum': 0.9,
          'weightDecay': 1e-4, 'printfreq': 10, 'traindir': traindir, 'valdir': valdir,
          'evaluate': False, 'resume': resumedir, 'bestmodel': bestmodeldir}