import torch.nn as nn
import torch
from Networks.NetworkArchitectures import createNetwork

def getNaiveAndLipltConstants(net, args, networkConfiguration, device):
    # This function calculates both the naive Lipschitz constant and the Lipschitz constant of LipLT.
    # If the original model is "LipLT", it can calculate the naive Lipschitz as well. However, if the original model
    # is "naive", it cannot calculate the Lipschitz constant of LipLT. Therefore, we need to create a new model.
    # This function calculates and returns both Lipschitz constants.
    if args.modelLipschitzType == "liplt":
        if isinstance(net, nn.DataParallel):
            improvedLipschitz = net.module.calculateNetworkLipschitz(0)
            temp = net.module.numberOfPowerIterations
            net.module.numberOfPowerIterations = 10
            naiveLipschitz = net.module.calculateNetworkLipschitz(0, True)
            net.module.numberOfPowerIterations = temp
        else:
            improvedLipschitz = net.calculateNetworkLipschitz(0)
            temp = net.numberOfPowerIterations
            net.numberOfPowerIterations = 10
            naiveLipschitz = net.calculateNetworkLipschitz(0, True)
            net.numberOfPowerIterations = temp
    elif args.modelLipschitzType == "naive":
        if isinstance(net, nn.DataParallel):
            naiveLipschitz = net.module.calculateNetworkLipschitz(0)
            stateDictionary = net.module.state_dict()
        else:
            naiveLipschitz = net.calculateNetworkLipschitz(0)
            stateDictionary = net.state_dict()
        networkConfiguration['modelType'] = "liplt"
        loadEigenVectors = False
        networkConfiguration['numberOfPowerIterations'] = 100
        net2 = createNetwork(networkConfiguration, device)
        net2.to(device)
        net2.eval()

        net2.load_state_dict(stateDictionary, loadEigenVectors=loadEigenVectors)
        improvedLipschitz = net2.calculateNetworkLipschitz()
        networkConfiguration['numberOfPowerIterations'] = args.numberOfPowerIterations
        networkConfiguration['modelType'] = args.modelLipschitzType
    else:
        raise ValueError("modelLipschitzType is not valid.")
    return naiveLipschitz, improvedLipschitz

def updateEigenVectorsForLipschitz(net, modelLipschitzType):
    # This function is used to keep the eigen vectors of the network updated. This is necessary for calculating the
    # Lipschitz constant at the end. This function is useful in particular for when the training procedure itself does
    # not update the eigenvectors, e.g., in standard training.
    if modelLipschitzType == "liplt":
        if isinstance(net, nn.DataParallel):
            _ = net.module.calculateNetworkLipschitz()
            _ = net.module.calculateNetworkLipschitz(0, True)
        else:
            _ = net.calculateNetworkLipschitz()
            _ = net.calculateNetworkLipschitz(0, True)
    elif modelLipschitzType == "naive":
        if isinstance(net, nn.DataParallel):
            _ = net.module.calculateNetworkLipschitz()
        else:
            _ = net.calculateNetworkLipschitz()
