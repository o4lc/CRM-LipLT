import datetime
import sys
import os
import time

import numpy as np
from typing import List
import warnings
import argparse

import torch.cuda

import wandb
import test
from Utilities.Schedulers import createScheduler, getSchedulerConfiguration
from Utilities.Accuracies import *
from Utilities.LipschitzUtility import *
from data.DataLoader import createDataLoaders
from Utilities.Perturbations import *
from Losses.LossConfigurations import getCriterionConfiguration, createCriterion


torch.autograd.set_detect_anomaly(True)
from Networks.NetworkArchitectures import getNetworkArchitecture, createNetwork
# from torchviz import make_dot, make_dot_from_trace



def createOptimizer(optimizerConfiguration, network):
    if optimizerConfiguration['type'] == "Adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=optimizerConfiguration['lr'],
                               betas=optimizerConfiguration['betas'],
                               amsgrad=optimizerConfiguration['amsgrad'],
                               eps=optimizerConfiguration['eps'])
    else:
        raise NotImplementedError
    # optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    return optimizer


def createCheckpointDictionary(checkpointDictionary, currentEpoch,
                               bestStateDictionary, bestLoss, networkStateDictionary,
                               optimizerStateDictionary, schedulerStateDictionary,
                               criterionStateDictionary):
    checkpointDictionary["optimizerStateDictionary"] = optimizerStateDictionary
    checkpointDictionary['schedulerStateDictionary'] = schedulerStateDictionary
    checkpointDictionary['networkStateDictionary'] = networkStateDictionary
    checkpointDictionary['bestStateDictionary'] = bestStateDictionary
    checkpointDictionary['bestLoss'] = bestLoss
    checkpointDictionary['currentEpoch'] = currentEpoch
    checkpointDictionary['criterionStateDictionary'] = criterionStateDictionary

    return checkpointDictionary



def generateNameFromArgs(args):
    def addString(valueName, value):
        return name + "_" + valueName +  "_" + str(value)
    name = args.projectName
    name = addString("test_eps", args.test_eps)
    name = addString("train_eps", args.train_eps)
    name = addString("dataset", args.dataset)
    name = addString("criterion", args.criterionType)
    name = addString("seed", args.randomSeed)
    return name

def saveStringToFile(stringToWrite, accuracySaveDirectory, fileName):
    if not os.path.exists(accuracySaveDirectory):
        os.makedirs(accuracySaveDirectory)
    # open file and write to it
    with open(accuracySaveDirectory + fileName, "a") as file:
        file.write(stringToWrite)

def main(args):

    assert args.lipschitzNorm == 2
    torch.manual_seed(args.randomSeed)
    identifier = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")

    if args.wandb:
        wandb.init(project="", entity="", config=args, name=generateNameFromArgs(args))
    else:
        wandb.init(project="lipschitz", config=args, name=generateNameFromArgs(args), mode="disabled")
    validationStep = args.validationStep
    model_name = args.projectName

    device = args.device

    if device.startswith("cpu"):
        device = torch.device(device)
    elif device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine.")
    else:
        if args.multiGpu:
            device = torch.device("cuda")
        else:
            device = torch.device(device)


    checkpointResume = args.checkpointResume
    if checkpointResume:
        warnings.warn("Resuming from checkpoint.")
    # checkpointLocation = "D:/Dataset/Lipchitz/"
    checkpointLocation = args.checkpointLocation
    saveFileDirectory = args.saveFileDirectory
    if not os.path.exists(checkpointLocation):
        os.makedirs(checkpointLocation)
    if not os.path.exists(saveFileDirectory):
        os.makedirs(saveFileDirectory)

    if checkpointResume:
        checkpointDictionary = torch.load(checkpointLocation + model_name)
        smoothing = checkpointDictionary['smoothing']
        smoothingEpsilon = checkpointDictionary['smoothingEpsilon']
        test_eps = checkpointDictionary['test_eps']
        train_eps = checkpointDictionary['train_eps']
        inputPerturbationNorm = checkpointDictionary['inputPerturbationNorm']
        lipschitzNorm = checkpointDictionary['lipschitzNorm']
        dataset = checkpointDictionary['dataset']
        numberOfWorkers = checkpointDictionary['numberOfWorkers']
        batchSize = checkpointDictionary['batchSize']
        validationSplit = checkpointDictionary['validationSplit']
        maxEpoch = checkpointDictionary['maxEpoch']
        currentEpoch = checkpointDictionary['currentEpoch']
        trainLosses = checkpointDictionary['trainLosses']
        validationLosses = checkpointDictionary['validationLosses']
        validationAccuracies = checkpointDictionary['validationAccuracies']
        bestStateDictionary = checkpointDictionary['bestStateDictionary']
        bestLoss = checkpointDictionary['bestLoss']
        dataLoaders = checkpointDictionary['dataLoaders']
        # warnings.warn("Creating a new dataset. Validation accuracies will not be correct")
        # dataLoaders = createDataLoaders(dataset, validationSplit, batchSize, numberOfWorkers)
        architectureToUse = checkpointDictionary['architectureToUse']
        regularizeMisclassified = checkpointDictionary['regularizeMisclassified']

        networkConfiguration = checkpointDictionary['networkConfiguration']
        optimizerConfiguration = checkpointDictionary['optimizerConfiguration']
        schedulerConfiguration = checkpointDictionary['schedulerConfiguration']
        criterionConfiguration = checkpointDictionary['criterionConfiguration']
        imageHeight = checkpointDictionary['imageHeight']
        imageWidth = checkpointDictionary['imageWidth']
        numberOfClasses = checkpointDictionary['numberOfClasses']

        networkStateDict = checkpointDictionary['networkStateDictionary']
        optimizerStateDict = checkpointDictionary['optimizerStateDictionary']
        schedulerStateDict = checkpointDictionary['schedulerStateDictionary']
        criterionStateDict = checkpointDictionary['criterionStateDictionary']

        net = createNetwork(networkConfiguration, device)
        net.load_state_dict(networkStateDict)
        net.to(device)
        optimizer = createOptimizer(optimizerConfiguration, net)
        optimizer.load_state_dict(optimizerStateDict)
        scheduler = createScheduler(schedulerConfiguration, optimizer)
        scheduler.load_state_dict(schedulerStateDict)
        criterion = createCriterion(criterionConfiguration)
        criterion.load_state_dict(criterionStateDict)

    else:
        startPretrained = args.startPretrained
        pretrainedFilePath = args.pretrainedFilePath
        smoothing = args.smoothing
        smoothingEpsilon = args.smoothingEpsilon
        test_eps = args.test_eps
        train_eps = args.train_eps
        inputPerturbationNorm = args.inputPerturbationNorm
        lipschitzNorm = args.lipschitzNorm


        dataset = args.dataset
        numberOfWorkers = args.numberOfWorkers
        batchSize = args.batchSize
        validationSplit = args.validationSplit
        regularizeMisclassified = args.regularizeMisclassified

        maxEpoch = args.maxEpoch
        if maxEpoch <= 10:
            warnings.warn('maxEpoch set small for test only')
        if dataset == "MNIST":
            imageHeight = imageWidth = 28
            numberOfClasses = 10
        elif dataset == "CIFAR10":
            imageHeight = imageWidth = 32
            numberOfClasses = 10
        elif dataset == "CIFAR100":
            imageHeight = imageWidth = 32
            numberOfClasses = 100
        elif dataset == "tiny-imagenet":
            imageHeight = imageWidth = 64
            numberOfClasses = 200
        else:
            raise ValueError
        args.numberOfClasses = numberOfClasses

        dataLoaders = createDataLoaders(args=args)
        optimizerConfiguration = {"type": "Adam", "lr": args.initialLearningRate, "betas": (0.9, 0.999),
                                  "amsgrad": False, "eps": 1e-7}

        criterionConfiguration = getCriterionConfiguration(args.criterionType, args)
        schedulerConfiguration = getSchedulerConfiguration(args.schedulerType, args)

        architectureToUse = args.architecture
        networkConfiguration = {"perClassLipschitz": args.perClassLipschitz,
                                "activation": "relu",
                                "numberOfPowerIterations": args.numberOfPowerIterations,
                                "inputShape": dataLoaders['train'].dataset[0][0].shape,
                                "modelType": args.modelLipschitzType,
                                'layers': getNetworkArchitecture(architectureToUse),
                                "weightInitialization": args.weightInitialization,
                                'pairwiseLipschitz': args.pairwiseLipschitz,
                                'architecture': architectureToUse,
                                'numberOfClasses': numberOfClasses,}

        currentEpoch = 0
        trainLosses = []
        validationLosses = []
        validationAccuracies = [0.0]
        bestStateDictionary = None
        bestLoss = 10000

        net = createNetwork(networkConfiguration, device)
        optimizer = createOptimizer(optimizerConfiguration, net)
        scheduler = createScheduler(schedulerConfiguration, optimizer)
        criterion = createCriterion(criterionConfiguration)

        if startPretrained:
            assert dataset in pretrainedFilePath and architectureToUse in pretrainedFilePath
            warnings.warn("Loading pretrained model from " + pretrainedFilePath)
            net.load_state_dict(torch.load(pretrainedFilePath, map_location=device)['stateDictionary'],
                                loadEigenVectors=False)

        checkpointDictionary = {
            "startPretrained": startPretrained,
            "pretrainedFilePath": pretrainedFilePath,
            "identifier": identifier,
            "smoothing": smoothing,
            "smoothingEpsilon": smoothingEpsilon,
            "test_eps": test_eps,
            "train_eps": train_eps,
            "inputPerturbationNorm": inputPerturbationNorm,
            "lipschitzNorm": lipschitzNorm,
            "dataset": dataset,
            'numberOfClasses': numberOfClasses,
            "numberOfWorkers": numberOfWorkers,
            "batchSize": batchSize,
            "validationSplit": validationSplit,
            "maxEpoch": maxEpoch,
            "trainLosses": trainLosses,
            "validationLosses": validationLosses,
            'validationAccuracies': validationAccuracies,
            "dataLoaders": dataLoaders,
            "architectureToUse": architectureToUse,
            "networkConfiguration": networkConfiguration,
            "optimizerConfiguration": optimizerConfiguration,
            "schedulerConfiguration": schedulerConfiguration,
            "criterionConfiguration": criterionConfiguration,
            "imageHeight": imageHeight,
            "imageWidth": imageWidth,
            "regularizeMisclassified": regularizeMisclassified,
        }

    if inputPerturbationNorm != lipschitzNorm:
        print("Norm of input perturbation is not the same as the lipschitz norm.\n"
              " Have you taken this into consideration?")
        time.sleep(5)
        print("If you say so ...")
    # crossEntropyCriterion = createCriterion({"type": "standard"})

    improvedLipschitzConstantsOverTraining = []
    naiveLipschitzConstantsOverTraining = []
    if args.multiGpu:
        multiGpuDevices = args.multiGpuDevices
    elif args.device.startswith("cuda"):
        multiGpuDevices = [int(args.device.split(":")[-1])]
    if device.type == "cuda":
        net = nn.DataParallel(net, device_ids=multiGpuDevices)
    net.to(device)
    iterationTimes = []
    lipschitzTimes = []
    startTime = time.time()
    for currentEpoch in range(currentEpoch, maxEpoch):
        wandb.log({"epoch": currentEpoch})
        if np.mod(currentEpoch, validationStep) == 0:
            loopPhases = ["train", 'eval']
        else:
            loopPhases = ["train"]

        for phase in loopPhases:
            if phase == "train":
                eps = train_eps
                net.train()
            else:
                eps = test_eps
                net.eval()
                tempValidationLosses = []
                tempValidationAccuracies = []
            startTime = time.time()

            for i, (X, Y) in enumerate(dataLoaders[phase]):
                if args.timeEvents:
                    # Accurate timing using CUDA events
                    epochStartEvent = torch.cuda.Event(enable_timing=True)
                    epochEndEvent = torch.cuda.Event(enable_timing=True)
                    lipschitzStartEvent = torch.cuda.Event(enable_timing=True)
                    lipschitzEndEvent = torch.cuda.Event(enable_timing=True)
                    epochStartEvent.record()

                X = X.to(device)
                Y = Y.to(device)
                originalY = Y
                if smoothing:
                    Y = smoothLabels(Y, numberOfClasses, smoothingEpsilon)

                out = net(X)
                maxIndices = torch.argmax(out, 1)

                thisBatchSize = X.shape[0]
                optimizer.zero_grad()

                if criterionConfiguration['type'] == "standard":
                    loss = criterion(out, Y)
                elif criterionConfiguration['type'] == "GloRo":
                    lipschitzConstants = net.calculateNetworkLipschitz()
                    perturbation = createGloroPerturbation(lipschitzConstants, thisBatchSize, Y, eps)
                    modifiedOutput = modifyOutputForVerifiedAccuracy(out, perturbation, thisBatchSize, Y,
                                                                     regularizeOnlyIfCorrectlyClassified=False)
                    wandb.log({"average lipschitz constant": torch.mean(lipschitzConstants).item()})
                    loss = criterion(modifiedOutput, Y)
                elif criterionConfiguration['type'] == "LMT":
                    lipschitzConstants = net.calculateNetworkLipschitz()
                    wandb.log({"lipschitz constant": lipschitzConstants.mean().item()})
                    perturbation = createLmtPerturbation(lipschitzConstants, thisBatchSize, numberOfClasses, Y, eps)
                    perturbation[maxIndices != Y, :] = 0
                    loss = criterion(out, Y, perturbation)
                elif criterionConfiguration['type'] in ['certifiedRadiusMaximization']:
                    # posits = nn.functional.softmax(out, 1)
                    posits = out
                    margins = posits[range(thisBatchSize), maxIndices].unsqueeze(1) - posits
                    if args.timeEvents:
                        lipschitzStartEvent.record()
                    if isinstance(net, nn.DataParallel):
                        lipschitzConstants = net.module.calculateNetworkLipschitz(selfPairDefaultValue=1)
                        pairWiseLipschitzConstants = net.module.createPairwiseLipschitzFromLipschitz(lipschitzConstants, numberOfClasses, 1)
                    else:
                        lipschitzConstants = net.calculateNetworkLipschitz(selfPairDefaultValue=1)
                        pairWiseLipschitzConstants = net.createPairwiseLipschitzFromLipschitz(lipschitzConstants,
                                                                                                     numberOfClasses, 1)
                    if args.timeEvents:
                        torch.cuda.synchronize()
                        lipschitzEndEvent.record()
                        torch.cuda.synchronize()
                        lipschitzTimes.append(lipschitzStartEvent.elapsed_time(lipschitzEndEvent))
                    certifiedRadii = margins / pairWiseLipschitzConstants[maxIndices, :]
                    certifiedRadii[range(thisBatchSize), maxIndices] = torch.inf
                    if not regularizeMisclassified:
                        certifiedRadii = certifiedRadii[maxIndices == originalY, :]

                    loss = criterion(out, Y, certifiedRadii, verbose=i==0)

                if phase == "train":
                    wandb.log({"train loss": loss.cpu().item()})
                    loss.backward()
                    optimizer.step()
                    # This step is necessary for the Lipschitz constant calculation
                    if isinstance(net, nn.DataParallel):
                        net.module.miniBatchStep()
                    else:
                        net.miniBatchStep()
                else:
                    tempValidationLosses.append(loss.cpu().item())
                    if smoothing:
                        Y = torch.argmax(Y, 1)
                    tempValidationAccuracies.append(sum(maxIndices == Y).cpu().detach() / thisBatchSize * 100)
                    if loss < bestLoss:
                        bestLoss = loss.item()
                        torch.save(net.state_dict(), checkpointLocation + "tempBestStateDict.pth")

                if args.logLipschitzConstants:
                    with torch.no_grad():
                        updateEigenVectorsForLipschitz(net, args.modelLipschitzType)

                torch.cuda.empty_cache()
                if args.timeEvents:
                    torch.cuda.synchronize()
                    epochEndEvent.record()
                    torch.cuda.synchronize()
                    iterationTimes.append(epochStartEvent.elapsed_time(epochEndEvent))
            endTime = time.time()
            epochTime = endTime - startTime
            print("epoch time: ", epochTime)
            if phase == "train":
                if args.timeEvents:
                    print("average iteration time", np.mean(iterationTimes),
                          "std", np.std(iterationTimes))
                    print("average time for lipschitz calculation", np.mean(lipschitzTimes),
                          "std", np.std(lipschitzTimes))

                criterion.schedulerStep()
                scheduler.step()
                trainLosses.append(loss.item())

                checkpointDictionary =\
                    createCheckpointDictionary(checkpointDictionary, currentEpoch, bestStateDictionary, bestLoss,
                                               net.module.state_dict() if isinstance(net, nn.DataParallel)
                                               else net.state_dict(),
                                               optimizer.state_dict(), scheduler.state_dict(),
                                               criterion.state_dict())
                torch.save(checkpointDictionary, checkpointLocation + model_name)
            else:
                validationLosses.append(np.mean(tempValidationLosses))
                validationAccuracies.append(np.mean(tempValidationAccuracies))
                wandb.log({"validation loss": np.mean(validationLosses[-1]),
                           "validation accuracy": validationAccuracies[-1]})
            if np.mod(currentEpoch, validationStep) == 0:
                if phase == "train":
                    print('epoch:', currentEpoch)
                    print('Training loss: ', loss.item())
                else:
                    print("validation losses average: ", validationLosses[-1])
                    calculateModelAccuracy(net, dataLoaders, device, "eval")
                    verifiedAccuracy = calculateVerifiedAccuracy(net, dataLoaders, eps, criterionConfiguration['type'],
                                                                 numberOfClasses, device, "eval")
                    wandb.log({"verified accuracy": verifiedAccuracy})
        if args.logLipschitzConstants:
            with torch.no_grad():
                naiveLipschitz, improvedLipschitz = getNaiveAndLipltConstants(net, args, networkConfiguration, device)
                improvedLipschitzConstantsOverTraining.append(improvedLipschitz.detach().cpu().numpy())
                naiveLipschitzConstantsOverTraining.append(naiveLipschitz.detach().cpu().numpy())
    endTime = time.time()
    print("Time to train: {}".format(endTime - startTime))
    if args.logLipschitzConstants:
        torch.save({"improvedLipschitzConstantsOverTraining": improvedLipschitzConstantsOverTraining,
                    "naiveLipschitzConstantsOverTraining": naiveLipschitzConstantsOverTraining},
                   checkpointLocation + model_name + '_' +
                   "lipschitzConstantsOverTraining" + '-' + networkConfiguration["modelType"] +
                   '_' + identifier + ".pth")
    if args.timeEvents:
        print("average iteration time", np.mean(iterationTimes),
              "std", np.std(iterationTimes))
        print("average time for lipschitz calculation", np.mean(lipschitzTimes),
              "std", np.std(lipschitzTimes))

        stringToWrite = "Architecture: {}, Model type: {}, Number of power iterations: {}, Criterion : {}\n"\
            .format(architectureToUse, args.modelLipschitzType, args.numberOfPowerIterations, criterionConfiguration['type'])
        stringToWrite += "average iteration time. Then average time for Lipschitz calculation:" \
                         "\n {} std: {}\n" \
                         " {} std: {}\n".format(np.mean(iterationTimes),
                                                np.std(iterationTimes),
                                                np.mean(lipschitzTimes),
                                                np.std(lipschitzTimes))
        accuracySaveDirectory = "scriptResults/"
        if not os.path.exists(accuracySaveDirectory):
            os.makedirs(accuracySaveDirectory)
        timeFileName = args.timeFileName
        # open file and write to it
        with open(accuracySaveDirectory + timeFileName, "a") as file:
            file.write(stringToWrite)

    # save now. We'll update the saved file after the accuracies are acquired as well
    saveFileName = saveFileDirectory + model_name + '_' +\
                   checkpointDictionary['dataset'] + '_' +\
                   networkConfiguration["modelType"] + '_' +\
                   criterionConfiguration["type"] + '_' + \
                   identifier + ".pth"
    torch.save({"stateDictionary": net.state_dict(), "checkpointDictionary": checkpointDictionary},
               saveFileName)
    net.eval()

    modelAccuracy, verifiedAccuracy, pgdAccuracy, lipschitzConstants = test.test(checkpointDictionary, device)

    torch.save({"stateDictionary": net.state_dict(), "checkpointDictionary": checkpointDictionary,
                "standardAccuracy": modelAccuracy, "verifiedAccuracy": verifiedAccuracy, "pgdAccuracy": pgdAccuracy,
                "lipschitzConstants": lipschitzConstants},
               saveFileName)

    if args.saveAccuracies and criterionConfiguration['type'] == "certifiedRadiusMaximization":
        stringToWrite = (f"Architecture: {architectureToUse},"
                         f" Model type: {args.modelLipschitzType},"
                         f" Number of power iterations: {args.numberOfPowerIterations}\n")
        stringToWrite += (f"Lambda {criterionConfiguration['initialMu']},"
                          f" Alpha {criterionConfiguration['alpha']},"
                          f" R0 {criterionConfiguration['maximumPenalizingRadius']}\n")
        stringToWrite += (f"Standard accuracy: {modelAccuracy}\t,"
                          f" Verified accuracy: {verifiedAccuracy}\t,"
                          f" PGD accuracy: {pgdAccuracy}\n")
        saveStringToFile(stringToWrite, "scriptResults/", args.accuracyFileName)

def pureLipschitzCalculation(args):
    device = args.device
    maxEpoch = args.maxEpoch
    dataset = args.dataset
    if dataset in ["MNIST", "CIFAR10"]:
        numberOfClasses = 10
    elif dataset == "CIFAR100":
        numberOfClasses = 100
    elif dataset == "tiny-imagenet":
        numberOfClasses = 200
    else:
        raise ValueError
    dataLoaders = createDataLoaders(args=args)
    networkConfiguration = {"perClassLipschitz": args.perClassLipschitz,
                            "activation": "relu",
                            "numberOfPowerIterations": args.numberOfPowerIterations,
                            "inputShape": dataLoaders['train'].dataset[0][0].shape,
                            "modelType": args.modelLipschitzType,
                            'layers': getNetworkArchitecture(args.architecture),
                            "weightInitialization": args.weightInitialization,
                            'pairwiseLipschitz': args.pairwiseLipschitz,
                            'architecture': args.architecture,
                            'numberOfClasses': numberOfClasses, }
    net = createNetwork(networkConfiguration, device)
    if args.multiGpu:
        multiGpuDevices = args.multiGpuDevices
    elif args.device.startswith("cuda"):
        multiGpuDevices = [int(args.device.split(":")[-1])]
    if device.type == "cuda":
        net = nn.DataParallel(net, device_ids=multiGpuDevices)
    net.to(device)
    lipschitzTimes = []
    with torch.no_grad():
        for _ in range(maxEpoch):
            lipschitzStartEvent = torch.cuda.Event(enable_timing=True)
            lipschitzEndEvent = torch.cuda.Event(enable_timing=True)


            if args.timeEvents:
                lipschitzStartEvent.record()
            if isinstance(net, nn.DataParallel):
                lipschitzConstants = net.module.calculateNetworkLipschitz(selfPairDefaultValue=1)
                pairWiseLipschitzConstants = net.module.createPairwiseLipschitzFromLipschitz(lipschitzConstants,
                                                                                             numberOfClasses, 1)
            else:
                lipschitzConstants = net.calculateNetworkLipschitz(selfPairDefaultValue=1)
                pairWiseLipschitzConstants = net.createPairwiseLipschitzFromLipschitz(lipschitzConstants,
                                                                                      numberOfClasses, 1)

            torch.cuda.synchronize()
            lipschitzEndEvent.record()
            torch.cuda.synchronize()
            lipschitzTimes.append(lipschitzStartEvent.elapsed_time(lipschitzEndEvent))
    print("average time for lipschitz calculation", np.mean(lipschitzTimes),
          "std", np.std(lipschitzTimes))

    stringToWrite = "Architecture: {}, Model type: {}, Number of power iterations: {}\n" \
        .format(args.architecture, args.modelLipschitzType, args.numberOfPowerIterations)
    stringToWrite += "Average time for Lipschitz calculation:" \
                     " {} std: {}\n".format(np.mean(lipschitzTimes),
                                            np.std(lipschitzTimes))
    saveStringToFile(stringToWrite, "scriptResults/", args.timeFileName)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--randomSeed', type=int, default=1)
    parser.add_argument('--validationStep', type=int, default=5, help="How often to validate the model")
    parser.add_argument('--projectName', type=str, default="test")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpointResume', action='store_true', default=False)
    parser.add_argument('--checkpointLocation', type=str, default="Networks/")
    parser.add_argument('--saveFileDirectory', type=str, default="Networks/FinalModels/")
    parser.add_argument('--startPretrained', action='store_true', default=False)
    parser.add_argument('--pretrainedFilePath', type=str, default="Networks/PretrainedModels/CIFAR10_6C2F.pth")
    parser.add_argument('--smoothing', action='store_true', default=False)
    parser.add_argument('--smoothingEpsilon', type=float, default=0.09)
    parser.add_argument('--test_eps', type=float, default=0.141)
    parser.add_argument('--train_eps', type=float, default=0.1551)
    parser.add_argument('--inputPerturbationNorm', type=int, default=2)
    parser.add_argument('--lipschitzNorm', type=int, default=2)
    parser.add_argument('--initialLearningRate', type=float, default=1e-3)
    parser.add_argument('--learningRateDecayEpoch', type=int, default=100)
    parser.add_argument('--smallestLearningRate', type=float, default=1e-6)

    parser.add_argument('--dataset', type=str, default="CIFAR10",
                        choices=["MNIST", "CIFAR10",  "CIFAR100", "tiny-imagenet"])
    parser.add_argument('--datasetAugmentationDegree', type=int, default=5)
    parser.add_argument('--datasetAugmentationTranslation', type=float, default=0.1)
    parser.add_argument('--numberOfWorkers', type=int, default=2)
    parser.add_argument('--batchSize', type=int, default=512)
    parser.add_argument('--validationSplit', type=float, default=0.9)
    parser.add_argument('--maxEpoch', type=int, default=400)

    parser.add_argument('--criterionType', type=str, default="certifiedRadiusMaximization",
                        choices=["standard", "GloRo", "LMT", "certifiedRadiusMaximization"])
    parser.add_argument('--schedulerType', type=str, default="exponentialWithStopping")
    parser.add_argument('--architecture', type=str, default="6C2F",
                        help="Architecture to use. Check the architectures file for more help")
    parser.add_argument('--perClassLipschitz', action='store_true', default=False)
    parser.add_argument('--pairwiseLipschitz', action='store_true', default=False)
    parser.add_argument('--numberOfPowerIterations', type=int, default=1)
    parser.add_argument('--weightInitialization', type=str, default="standard")
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--modelLipschitzType', type=str, default="liplt",
                        choices=["liplt", "naive"])

    # Criterion parameters
    parser.add_argument("--robustStartingEpoch", type=int, default=0)

    parser.add_argument('--gloroInitialLambda', type=float, default=0.1)

    parser.add_argument("--lmtInitialLambda", type=float, default=0.001, help="Initial lambda for LMT")
    parser.add_argument('--lmtLambdaGamma', type=float, default=1.03, help="Gamma scheduler for the Lambda of LMT")
    parser.add_argument('--lmtMaximumLambda', type=float, default=2, help="Maximum lambda for LMT")

    parser.add_argument('--radiusMaximizationInitialLambda', type=float, default=50)
    parser.add_argument('--radiusMaximizationLambdaGamma', type=float, default=1.)
    parser.add_argument('--radiusMaximizationMaximumLambda', type=float, default=1000)
    parser.add_argument('--radiusMaximizationMinimumLambda', type=float, default=0.)
    parser.add_argument('--radiusMaximizationAlpha', type=float, default=3)
    parser.add_argument('--radiusMaximizationMaximumPenalizingRadius', type=float, default=0.5)
    parser.add_argument('--radiusMaximizationMaximumPenalizingRadiusGamma', type=float, default=1)
    parser.add_argument('--radiusMaximizationMaximumPenalizingRadiusMaximumValue', type=float, default=100)
    parser.add_argument('--radiusMaximizationCrossEntropyType', type=str, default="crossEntropy",
                        choices=["crossEntropy", "boostedCrossEntropy"])
    parser.add_argument('--radiusMaximizationRobustLossType', type=str, default="max",
                        choices=['max', 'softMax'])

    parser.add_argument('--multiGpu', action='store_true', default=False)
    parser.add_argument('--multiGpuDevices', nargs='+', default=[0], type=int)

    parser.add_argument('--regularizeMisclassified', action='store_true', default=False)

    parser.add_argument('--logLipschitzConstants', action='store_true', default=False)
    parser.add_argument('--timeEvents', action='store_true', default=False)
    parser.add_argument('--timeFileName', type=str, default="timeDepth.txt")
    parser.add_argument('--saveAccuracies', action='store_true', default=False)
    parser.add_argument('--accuracyFileName', type=str, default="accuracy.txt")

    parser.add_argument('--wandb', action='store_true', default=False,
                        help="Use wandb for logging. By default, wandb is deactivated."
                             " To use it, please fix the relevant project and entity in the initialization")
    parser.add_argument('--pureLipschitzCalculation', action='store_true', default=False)
    ##
    args = parser.parse_args()
    if args.pureLipschitzCalculation:
        pureLipschitzCalculation(args)
    else:
        main(args)





