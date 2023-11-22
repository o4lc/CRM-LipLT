import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import time

from data.DataLoader import createDataLoaders
from Utilities.Perturbations import *


def calculateModelAccuracy(net, dataset, device=torch.device("cuda"), datasetPart='test'):
    if type(dataset) == str:
        batchSize = 1024
        if "imagenet" in dataset:
            batchSize = 256
        dataset = createDataLoaders(dataset, batchSize=batchSize)[datasetPart]
    else:
        dataset = dataset[datasetPart]
    numberOfCorrect = 0
    for X, Y in dataset:
        X = X.to(device)
        Y = Y.to(device)
        out = net(X)
        maxIndices = torch.argmax(out, 1)
        # print(torch.sum(maxIndices == Y), Y.shape)
        numberOfCorrect += torch.sum(maxIndices == Y)
    # print(Y, maxIndices)
    modelAccuracy = numberOfCorrect / dataset.dataset.__len__() * 100
    print("Accuracy on {} dataset: {}".format(datasetPart, modelAccuracy))
    return modelAccuracy


def calculateVerifiedAccuracy(net, dataset, eps, verifiedAccuracyType, numberOfClasses, device=torch.device("cuda"),
                              datasetPart="test", verbose=True):
    if type(dataset) == str:
        batchSize = 1024
        if "imagenet" in dataset:
            batchSize = 256

        dataset = createDataLoaders(dataset, batchSize=batchSize)[datasetPart]
    else:
        dataset = dataset[datasetPart]
    if isinstance(net, nn.DataParallel):
        lipschitzConstants = net.module.calculateNetworkLipschitz()
    else:
        lipschitzConstants = net.calculateNetworkLipschitz()

    numberOfCorrect = 0
    for X, Y in dataset:
        X = X.to(device)
        Y = Y.to(device)
        out = net(X)
        batchSize = Y.shape[0]
        if verifiedAccuracyType in ["standard", "LMT", "certifiedRadiusMaximization"]:
            perturbation = createLmtPerturbation(lipschitzConstants, batchSize, numberOfClasses, Y, eps)
            modifiedOutput = out + perturbation
        elif verifiedAccuracyType in ["GloRo"]:
            perturbation = createGloroPerturbation(lipschitzConstants, batchSize,
                                                   Y, eps)
            modifiedOutput = modifyOutputForVerifiedAccuracy(out, perturbation, batchSize, Y,
                                                             regularizeOnlyIfCorrectlyClassified=False)
        else:
            raise NotImplementedError
        modifiedMaxIndices = torch.argmax(modifiedOutput, 1)
        numberOfCorrect += torch.sum(modifiedMaxIndices == Y)
    verifiedAccuracy = (numberOfCorrect / dataset.dataset.__len__() * 100).item()
    if verbose:
        print("Verified accuracy percentage on " + datasetPart + " dataset: {}".format(verifiedAccuracy))
    return verifiedAccuracy

def calculatePgdAccuracy(net, dataset, eps, lBall=2, device=torch.device("cuda"), verbose=True):
    if type(dataset) == str:
        lowerBound = 0
        upperBound = 1
        if "CIFAR10" in dataset:
            lowerBound = -1
        batchSize = 1024
        if "tiny-imagenet" in dataset:
            batchSize = 8
        dataset = createDataLoaders(dataset, batchSize=batchSize)['test']
    else:
        print("Warning. Don't know the bounds of the dataset. Assuming [0, 1]")
        dataset = dataset['test']

    accuracies = (1 - evaluate_madry(dataset, net, eps, False, device, lBall=lBall,
                                     lowerBound=lowerBound, upperBound=upperBound).item()) * 100
    if verbose:
        print("PGD accuracy percentage on test dataset: {}".format(accuracies))
    return accuracies


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


def _pgd(model, X, Y, epsilon, niters=100, alpha=0.001, lBall=torch.inf, lowerBound=0, upperBound=1):
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, Y)
    err = (out.data.max(1)[1] != Y.data).float().sum() / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters):
        # print(i, model(X_pgd))
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        out = model(X_pgd)
        out = out - out.mean(1, keepdim=True)
        # loss = nn.CrossEntropyLoss()(out, Y)
        loss = -(1 + nn.functional.softmax(out, dim=1)[range(Y.shape[0]), Y]).log().mean()
        # print(loss)
        loss.backward()
        eta = alpha * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        if lBall == torch.inf:
            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
        elif lBall == 2:
            y = X_pgd.data - X.data
            if len(y.shape) == 2:
                norm = torch.linalg.norm(y, 2, 1, keepdim=True)
            elif len(y.shape) == 3:
                norm = torch.linalg.norm(y, "fro", (1, 2), keepdim=True)
            elif len(y.shape) == 4:
                norm = torch.linalg.norm(y.reshape(y.shape[0], -1), 2, 1).reshape(y.shape[0], 1, 1, 1)
            else:
                raise NotImplementedError
            X_pgd = Variable(torch.clamp(X.data + epsilon * y / norm, lowerBound, upperBound), requires_grad=True)
            # print(X_pgd)
        elif lBall == 1:
            raise NotImplementedError
        else:
            raise ValueError

    err_pgd = (model(X_pgd).data.max(1)[1] != Y.data).float().sum() / X.size(0)
    return err, err_pgd


def evaluate_madry(loader, model, epsilon, verbose, device=torch.device("cuda"), lBall=torch.inf,
                   lowerBound=0, upperBound=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    perrors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)


        # # perturb
        _, pgd_err = _pgd(model, Variable(X), Variable(y), epsilon, lBall=lBall,
                          lowerBound=lowerBound, upperBound=upperBound)

        # print to logfile
        # print(epoch, i, ce.item(), err, file=log)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        perrors.update(pgd_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PGD Error {perror.val:.3f} ({perror.avg:.3f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                      i, len(loader), batch_time=batch_time, loss=losses,
                      error=errors, perror=perrors))
        # log.flush()
    if verbose:
        print(' * PGD error {perror.avg:.3f}\t'
              'Error {error.avg:.3f}'
              .format(error=errors, perror=perrors))
    return perrors.avg