import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import time

class Conv2dBatchNorm(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding, bias=True):
        super(Conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias=bias)
        self.batchNorm = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        return self.batchNorm(self.conv(x))

    def applyForwardPassOfPowerIteration(self, x):
        layer = self.conv
        x = nn.functional.conv2d(x, layer.weight, None, layer.stride, layer.padding)
        layer = self.batchNorm
        x = nn.functional.batch_norm(x, torch.zeros_like(layer.running_var), layer.running_var, layer.weight)
        return x

    def applyBackwardPassOfPowerIteration(self, x):
        layer = self.batchNorm
        x = nn.functional.batch_norm(x, torch.zeros_like(layer.running_var), layer.running_var, layer.weight)
        layer = self.conv
        x = nn.functional.conv_transpose2d(x, layer.weight, None, layer.stride, layer.padding)
        return x


class SequentialLipschitzNetwork(nn.Module):
    def __init__(self, layers, weightInitialization):
        super(SequentialLipschitzNetwork, self).__init__()
        if weightInitialization == "normal":
            for layer in layers:
                if hasattr(layer, "weight"):
                    n = 1
                    for size in layer.weight.shape:
                        n *= size
                    layer.weight.data.normal_(0, np.sqrt(2 / n))
                if hasattr(layer, "bias"):
                    layer.bias.data.zero_()
        elif weightInitialization != "standard":
            if weightInitialization == "orthogonal":
                initFunction = init.orthogonal_
            elif weightInitialization == "xavierUniform":
                initFunction = init.xavier_uniform_
            elif weightInitialization == "xavierNormal":
                initFunction = init.xavier_normal_
            elif weightInitialization == "kaimingUniform":
                initFunction = init.kaiming_uniform_
            elif weightInitialization == "kaimingNormal":
                initFunction = init.kaiming_normal_
            else:
                raise ValueError("Unknown weight initialization function: {}".format(weightInitialization))
            for layer in layers:
                if hasattr(layer, "weight"):
                    initFunction(layer.weight)
        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)

    def calculateNetworkLipschitz(self):
        raise NotImplementedError

    def miniBatchStep(self):
        raise NotImplementedError


class SequentialNaiveLipschitz(SequentialLipschitzNetwork):
    def __init__(self, layers, sampleInputShape, device=torch.device("cuda"), perClassLipschitz=True,
                 numberOfPowerIterations=1,
                 weightInitialization="standard",
                 pairwiseLipschitz=False,
                 **kwargs):
        super(SequentialNaiveLipschitz, self).__init__(layers, weightInitialization)
        self.eigenVectorDictionary = {}
        self.subNetworkDictionary = {}
        self.indexMap = []
        self.preFlattenShape = None
        self.perClassLipschitz = perClassLipschitz
        self.pairwiseLipschitz = pairwiseLipschitz
        self.numberOfPowerIterations = numberOfPowerIterations
        self.device = device
        self.layers = layers
        self.shapes = []
        self.numberOfClasses = layers[-1].weight.shape[0]
        self.pairwiseClassLimit = 101

        if perClassLipschitz and pairwiseLipschitz:
            raise ValueError("Cannot have both perClassLipschitz and pairwiseLipschitz")

        x = torch.randn([1] + [*sampleInputShape])
        index = 0
        self.flattenIndex = -1
        flattenLayer = None
        with torch.no_grad():
            for layer in layers:
                if type(layer) in [nn.modules.linear.Linear, nn.modules.conv.Conv2d, Conv2dBatchNorm]:
                    self.shapes.append(x.shape)
                    self.indexMap.append(index)
                    index += 1
                    x = layer(x)
                elif type(layer) == nn.modules.flatten.Flatten:
                    self.flattenIndex = index
                    flattenLayer = layer
                    self.preFlattenShape = x.shape[1:]
                    x = layer(x)
                    index += 1
                elif type(layer) == nn.modules.activation.ReLU\
                        or type(layer) == nn.modules.activation.Tanh:
                    index += 1
                else:
                    raise ValueError

        self.createDictionaries(self.shapes, self.flattenIndex, flattenLayer)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        originalStateDictionary = super().state_dict(destination, prefix, keep_vars)
        dictionary = {"original": originalStateDictionary,
                      "eigenVectors": self.eigenVectorDictionary}
        return dictionary

    def load_state_dict(self, state_dict,
                        strict: bool = True,
                        loadEigenVectors=True):
        super().load_state_dict(state_dict['original'], strict)
        if loadEigenVectors:
            warned = False
            for key in self.eigenVectorDictionary:
                if key in state_dict['eigenVectors']:
                    self.eigenVectorDictionary[key] = state_dict['eigenVectors'][key]
                elif not warned:
                    warnings.warn("Some eigen vectors were not loaded. They were not found in the state dictionary "
                                  "and will be initialized randomly.\n"
                                  "It is best to set the number of power iterations"
                                  " to a higher number to achieve more accurate results if needed.")
                    warned = True

    def createDictionaries(self, shapes, flattenIndex, flattenLayer):
        for i in range(len(self.indexMap)):
            self.createEigenvectorAndSubNetwork(i, i, shapes, flattenIndex, flattenLayer)

    def createEigenvectorAndSubNetwork(self, startIndex, endIndex, shapes, flattenIndex, flattenLayer):
        if endIndex == len(self.indexMap) - 1 and self.perClassLipschitz:
            for k in range(self.numberOfClasses):
                self.eigenVectorDictionary[(startIndex, endIndex, k)] = torch.randn(shapes[startIndex]).to(self.device)
                self.subNetworkDictionary[(startIndex, endIndex, k)] = []
                for l in range(startIndex, endIndex + 1):
                    if self.indexMap[startIndex] < flattenIndex \
                            and self.indexMap[l] > flattenIndex > self.indexMap[l - 1]:
                        self.subNetworkDictionary[(startIndex, endIndex, k)].append(flattenLayer)
                    self.subNetworkDictionary[(startIndex, endIndex, k)].append(self.linear[self.indexMap[l]])
                self.subNetworkDictionary[(startIndex, endIndex, k)].append(nn.Linear(self.numberOfClasses, 1))
                currentWeight = torch.zeros((1, self.numberOfClasses)).to(self.device)
                currentWeight[0, k] = 1.
                self.subNetworkDictionary[(startIndex, endIndex, k)][-1].weight.data = \
                    nn.Parameter(currentWeight, requires_grad=False)
                self.subNetworkDictionary[(startIndex, endIndex, k)][-1].bias.data = \
                    nn.Parameter(torch.zeros(self.numberOfClasses).to(self.device), requires_grad=False)
        else:
            self.eigenVectorDictionary[(startIndex, endIndex)] = torch.randn(shapes[startIndex]).to(self.device)
            self.subNetworkDictionary[(startIndex, endIndex)] = []
            for l in range(startIndex, endIndex + 1):
                if self.indexMap[startIndex] < flattenIndex < self.indexMap[l] and self.indexMap[l - 1] < flattenIndex:
                    self.subNetworkDictionary[(startIndex, endIndex)].append(flattenLayer)
                self.subNetworkDictionary[(startIndex, endIndex)].append(self.linear[self.indexMap[l]])

        if endIndex == len(self.indexMap) - 1 and self.pairwiseLipschitz:
            for k in range(self.numberOfClasses):
                for j in range(k + 1, self.numberOfClasses):
                    self.eigenVectorDictionary[(startIndex, endIndex, k, j)] = \
                        torch.randn(shapes[startIndex]).to(self.device)
                    self.subNetworkDictionary[(startIndex, endIndex, k, j)] = []
                    for l in range(startIndex, endIndex + 1):
                        if self.indexMap[startIndex] < flattenIndex \
                                and self.indexMap[l] > flattenIndex > self.indexMap[l - 1]:
                            self.subNetworkDictionary[(startIndex, endIndex, k, j)].append(flattenLayer)
                        self.subNetworkDictionary[(startIndex, endIndex, k, j)].append(self.linear[self.indexMap[l]])
                    self.subNetworkDictionary[(startIndex, endIndex, k, j)].append(nn.Linear(self.numberOfClasses, 1))
                    currentWeight = torch.zeros((1, self.numberOfClasses)).to(self.device)
                    currentWeight[0, k] = 1.
                    currentWeight[0, j] = -1.
                    self.subNetworkDictionary[(startIndex, endIndex, k, j)][-1].weight.data = \
                        nn.Parameter(currentWeight, requires_grad=False)
                    self.subNetworkDictionary[(startIndex, endIndex, k, j)][-1].bias.data = \
                        nn.Parameter(torch.zeros(self.numberOfClasses).to(self.device), requires_grad=False)

    def forwardOnLayers(self, x, layers):
        for layer in layers:
            if isinstance(layer, nn.modules.linear.Linear):
                x = x @ layer.weight.T
            elif isinstance(layer, nn.modules.conv.Conv2d):
                x = nn.functional.conv2d(x, layer.weight, None, layer.stride, layer.padding)
            elif isinstance(layer, nn.modules.flatten.Flatten):
                x = x.flatten(1)
            elif isinstance(layer, Conv2dBatchNorm):
                x = layer.applyForwardPassOfPowerIteration(x)
            else:
                raise NotImplementedError
        return x

    def backwardOnLayers(self, x, layers):
        for layer in reversed(layers):
            if isinstance(layer, nn.modules.linear.Linear):
                x = x @ layer.weight
            elif isinstance(layer, nn.modules.conv.Conv2d):
                x = nn.functional.conv_transpose2d(x, layer.weight, None, layer.stride, layer.padding)
            elif isinstance(layer, nn.modules.flatten.Flatten):
                x = x.reshape(x.shape[0], *self.preFlattenShape)
            elif isinstance(layer, Conv2dBatchNorm):
                x = layer.applyBackwardPassOfPowerIteration(x)
            else:
                raise NotImplementedError
        return x

    def calculateSubNetNaiveLipschitz(self, x, startIndex, endIndex, numberOfPowerIterations, k=None, j=None):
        if k is None:
            layers = self.subNetworkDictionary[(startIndex, endIndex)]
        elif j is None:
            layers = self.subNetworkDictionary[(startIndex, endIndex, k)]
        else:
            layers = self.subNetworkDictionary[(startIndex, endIndex, k, j)]

        for i in range(numberOfPowerIterations):
            x = self.forwardOnLayers(x, layers)
            x = self.backwardOnLayers(x, layers)
            norm = torch.linalg.norm(x, None, None)
            x = x / norm

        y = self.forwardOnLayers(x, layers)
        return torch.linalg.norm(y, None, None), x

    def calculateNetworkLipschitz(self, selfPairDefaultValue=0):
        if self.pairwiseLipschitz:
            currentLipschitz = 1
            for startIndex in range(len(self.indexMap) - 1):
                endIndex = startIndex
                x = self.eigenVectorDictionary[(startIndex, endIndex)]
                l, x = self.calculateSubNetNaiveLipschitz(x, startIndex, endIndex, self.numberOfPowerIterations)
                self.eigenVectorDictionary[(startIndex, endIndex)] = x
                currentLipschitz *= l

            return self.calculateLargeClassLipschitzConstant(currentLipschitz, selfPairDefaultValue)
            # startIndex = endIndex = len(self.indexMap) - 1
            # finalLipschitz = [[] for _ in range(self.numberOfClasses)]
            # for k in range(self.numberOfClasses):
            #     for j in range(self.numberOfClasses):
            #         if k == j:
            #             finalLipschitz[k].append(torch.tensor(selfPairDefaultValue).to(self.device))
            #         elif j < k:
            #             finalLipschitz[k].append(finalLipschitz[j][k])
            #         else:
            #             x = self.eigenVectorDictionary[(startIndex, endIndex, k, j)]
            #             l, x = self.calculateSubNetNaiveLipschitz(x, startIndex, endIndex, self.numberOfPowerIterations,
            #                                                       k, j)
            #             self.eigenVectorDictionary[(startIndex, endIndex, k, j)] = x
            #             finalLipschitz[k].append(currentLipschitz * l)
            #     finalLipschitz[k] = torch.hstack(finalLipschitz[k])
            # return torch.vstack(finalLipschitz)
        elif self.perClassLipschitz:
            currentLipschitz = 1
            for startIndex in range(len(self.indexMap) - 1):
                endIndex = startIndex
                x = self.eigenVectorDictionary[(startIndex, endIndex)]
                l, x = self.calculateSubNetNaiveLipschitz(x, startIndex, endIndex, self.numberOfPowerIterations)
                self.eigenVectorDictionary[(startIndex, endIndex)] = x
                currentLipschitz *= l
            startIndex = endIndex = len(self.indexMap) - 1

            finalLipschitz = []
            for k in range(self.numberOfClasses):
                x = self.eigenVectorDictionary[(startIndex, endIndex, k)]
                l, x = self.calculateSubNetNaiveLipschitz(x, startIndex, endIndex, self.numberOfPowerIterations, k)
                self.eigenVectorDictionary[(startIndex, endIndex, k)] = x
                finalLipschitz.append(currentLipschitz * l)

            return torch.hstack(finalLipschitz).unsqueeze(1)
        else:
            currentLipschitz = 1
            for startIndex in range(len(self.indexMap)):
                endIndex = startIndex
                x = self.eigenVectorDictionary[(startIndex, endIndex)]
                l, x = self.calculateSubNetNaiveLipschitz(x, startIndex, endIndex, self.numberOfPowerIterations)
                self.eigenVectorDictionary[(startIndex, endIndex)] = x
                currentLipschitz *= l
            return currentLipschitz

    @staticmethod
    def createPairwiseLipschitzFromLipschitz(lipschitzConstants, numberOfClasses, selfPairDefaultValue=0):
        if len(lipschitzConstants.shape) < 2:
            pairWiseLipschitzConstants = torch.ones(numberOfClasses, numberOfClasses, device=lipschitzConstants.device)
            pairWiseLipschitzConstants *= torch.tensor(2 ** 0.5) * lipschitzConstants
            pairWiseLipschitzConstants[range(numberOfClasses), range(numberOfClasses)] = \
                torch.tensor(selfPairDefaultValue * 1.0).to(lipschitzConstants.device)
        elif len(lipschitzConstants.shape) == 2 and lipschitzConstants.shape[1] == 1:
            pairWiseLipschitzConstants = lipschitzConstants.T.repeat(numberOfClasses, 1)
            pairWiseLipschitzConstants = pairWiseLipschitzConstants + lipschitzConstants
            pairWiseLipschitzConstants[range(numberOfClasses), range(numberOfClasses)] = \
                torch.tensor(selfPairDefaultValue * 1.0).to(lipschitzConstants.device)
        elif len(lipschitzConstants.shape) == 2:
            pairWiseLipschitzConstants = lipschitzConstants
        else:
            raise ValueError("Lipschitz constants have wrong shape")
        return pairWiseLipschitzConstants

    def miniBatchStep(self):  # required to run after every loss.backward()
        for key in self.eigenVectorDictionary:
            self.eigenVectorDictionary[key].detach_()

    def calculateLargeClassLipschitzConstant(self, lipschitzUntilPenultimate, selfPairDefaultValue=0):
        rangeIterator = range(self.numberOfClasses)
        finalLayerWeight = self.layers[self.indexMap[-1]].weight

        stackedDifference = \
            torch.vstack([finalLayerWeight - finalLayerWeight[i:i + 1, :] for i in rangeIterator])

        calculatedNorms = torch.linalg.norm(stackedDifference, 2, 1)

        pairWiseLipschitzConstants = calculatedNorms.reshape(self.numberOfClasses, self.numberOfClasses) \
                                     * lipschitzUntilPenultimate
        pairWiseLipschitzConstants[rangeIterator, rangeIterator] = torch.tensor(selfPairDefaultValue * 1.0).to(
            lipschitzUntilPenultimate.device)
        return pairWiseLipschitzConstants


class SequentialLipltLipschitz(SequentialNaiveLipschitz):
    def __init__(self, layers, sampleInputShape, device=torch.device("cuda"), perClassLipschitz=True,
                 numberOfPowerIterations=1,
                 weightInitialization="standard",
                 pairwiseLipschitz=False,
                 **kwargs):
        super(SequentialLipltLipschitz, self).__init__(layers, sampleInputShape, device, perClassLipschitz,
                                                       numberOfPowerIterations, weightInitialization, pairwiseLipschitz,
                                                       **kwargs)

        if self.numberOfClasses > self.pairwiseClassLimit and self.pairwiseLipschitz:
            warnings.warn("Paiwise Lipschitz calculation for more than {} classes is not completely supported. "
                          "The improved implementation  is only used until the penultimate layer.".format(self.pairwiseClassLimit))

        # Since the new implementation does not use these dictionaries, just make them empty
        # self.eigenVectorDictionary = {}
        # self.subNetworkDictionary = {}
        self.numberOfPairs = self.numberOfClasses * (self.numberOfClasses - 1) // 2
        self.upUntilPenultimate = 1
        if not perClassLipschitz and not pairwiseLipschitz:
            self.upUntilPenultimate = 0
        numberOfWeightLayers = len(self.indexMap)
        self.batchEigenVectorDictionary = []
        for i in range(numberOfWeightLayers - self.upUntilPenultimate):
            self.batchEigenVectorDictionary.append(
                torch.randn([numberOfWeightLayers - self.upUntilPenultimate - i] + [*self.shapes[i][1:]]).to(self.device))
        # adding the inputs for the last layer in the case of pairwise or perClass lipschitz constant
        if self.perClassLipschitz:
            self.batchEigenVectorDictionary.append(
                torch.zeros(self.numberOfClasses, self.numberOfClasses).to(self.device))

            for j in range(self.numberOfClasses):
                self.batchEigenVectorDictionary[-1][j, j] = 1

        elif self.pairwiseLipschitz and self.numberOfClasses <= self.pairwiseClassLimit:

            self.batchEigenVectorDictionary.append(
                torch.zeros(self.numberOfPairs, self.numberOfClasses).to(self.device))

            count = 0
            for j in range(self.numberOfClasses):
                for k in range(j + 1, self.numberOfClasses):
                    self.batchEigenVectorDictionary[-1][count, j] = 1
                    self.batchEigenVectorDictionary[-1][count, k] = -1
                    count += 1

        self.forwardContinuingIndices = [[] for _ in range(numberOfWeightLayers - self.upUntilPenultimate)]
        self.forwardFinishedIndices = [[] for _ in range(numberOfWeightLayers - self.upUntilPenultimate)]
        self.backwardFinishedIndices = []
        self.backwardContinuingIndices = []
        extra = 0
        for i in range(numberOfWeightLayers - self.upUntilPenultimate):
            for j in range(i, numberOfWeightLayers - self.upUntilPenultimate + extra):
                self.forwardContinuingIndices[i].append(1)
            extra = len(self.forwardContinuingIndices[i]) - i - 1
            self.forwardContinuingIndices[i] = (np.cumsum(self.forwardContinuingIndices[i]) - 1).tolist()
            for j in range(0, len(self.forwardContinuingIndices[i]), numberOfWeightLayers - self.upUntilPenultimate - i):
                self.forwardContinuingIndices[i].remove(j)
                self.forwardFinishedIndices[i].append(j)
            self.backwardFinishedIndices.append(
                [-1 - remainingIndex for remainingIndex in reversed(self.forwardFinishedIndices[i])])
            self.backwardContinuingIndices.append(
                [-1 - continuingIndex for continuingIndex in reversed(self.forwardContinuingIndices[i])])
        # for firstCounter, i in enumerate(range(len(self.indexMap) - 2, -1, -1)):
        #     self.backwardFinishedIndices.append([-1 - remainingIndex for remainingIndex in reversed(self.forwardFinishedIndices[firstCounter])])
        #     self.backwardContinuingIndices.append([-1 - continuingIndex for continuingIndex in reversed(self.forwardContinuingIndices[firstCounter])])

    def createDictionaries(self, shapes, flattenIndex, flattenLayer):
        super().createDictionaries(shapes, flattenIndex, flattenLayer)

    def load_state_dict(self, state_dict,
                        strict: bool = True,
                        loadEigenVectors=True):
        super().load_state_dict(state_dict, strict, loadEigenVectors=loadEigenVectors)
        if loadEigenVectors:
            try:
                warned = False
                for i in range(len(self.batchEigenVectorDictionary)):
                    if self.batchEigenVectorDictionary[i].shape == state_dict['batchEigenVectors'][i].shape:
                        self.batchEigenVectorDictionary[i] = state_dict['batchEigenVectors'][i]
                    elif not warned:
                        warnings.warn("The provided state dictionary has different shapes for the batch eigen vectors.\n"
                                      "This is simply a warning. "
                                      "The eigen vectors with shape mismatch will be initialized randomly.")
                        warned = True
            except:
                warnings.warn("The provided state dictionary does not have the batch eigen vectors.\n"
                              "This is simply a warning. But you should fix this for the given state dictionary:)"
                              "The eigen vectors will be initialized randomly.")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        dictionary = super().state_dict(destination, prefix, keep_vars)
        dictionary['batchEigenVectors'] = self.batchEigenVectorDictionary
        return dictionary

    def applyLayerForward(self, layer, x):
        if isinstance(layer, nn.modules.linear.Linear):
            x = x @ layer.weight.T
        elif isinstance(layer, nn.modules.conv.Conv2d):
            x = nn.functional.conv2d(x, layer.weight, None, layer.stride, layer.padding)
        elif isinstance(layer, nn.modules.flatten.Flatten):
            x = x.flatten(1)
        elif isinstance(layer, Conv2dBatchNorm):
            x = layer.applyForwardPassOfPowerIteration(x)
        else:
            raise NotImplementedError
        return x

    def applyLayerBackward(self, layer, x):
        if isinstance(layer, nn.modules.linear.Linear):
            x = x @ layer.weight
        elif isinstance(layer, nn.modules.conv.Conv2d):
            x = nn.functional.conv_transpose2d(x, layer.weight, None, layer.stride, layer.padding)
        elif isinstance(layer, nn.modules.flatten.Flatten):
            x = x.reshape(x.shape[0], *self.preFlattenShape)
        elif isinstance(layer, Conv2dBatchNorm):
            x = layer.applyBackwardPassOfPowerIteration(x)
        else:
            raise NotImplementedError
        return x

    def applyPowerIteration(self):
        for _ in range(self.numberOfPowerIterations):

            appliedFlatten = False
            previousX = None
            remainingXs = []
            for i in range(len(self.indexMap) - self.upUntilPenultimate):
                x = self.batchEigenVectorDictionary[i]
                if previousX is not None:
                    if self.indexMap[i] > self.flattenIndex and not appliedFlatten:
                        previousX = previousX.flatten(1)
                        appliedFlatten = True
                    x = torch.vstack([previousX, x])
                x = self.applyLayerForward(self.layers[self.indexMap[i]], x)
                remainingXs.append(x[self.forwardFinishedIndices[i], :])
                previousX = x[self.forwardContinuingIndices[i], :]
            popped = remainingXs.pop()

            appliedFlatten = False
            for firstCounter, i in enumerate(range(len(self.indexMap) - 1 - self.upUntilPenultimate, -1, -1)):

                if self.indexMap[i] < self.flattenIndex and not appliedFlatten:
                    x = x.reshape(x.shape[0], *self.preFlattenShape)
                    appliedFlatten = True
                if i < len(self.indexMap) - 1 - self.upUntilPenultimate:
                    popped = remainingXs.pop()
                    x = torch.vstack([popped, x])
                x = self.applyLayerBackward(self.layers[self.indexMap[i]], x)
                finishedBatches = self.backwardFinishedIndices[firstCounter]
                continuingIndices = self.backwardContinuingIndices[firstCounter]
                finishedX = x[finishedBatches, :]
                x = x[continuingIndices, :]
                norm = torch.linalg.norm(finishedX.flatten(1), 2, 1, True)
                while len(norm.shape) < len(finishedX.shape):
                    norm = norm.unsqueeze(-1)
                self.batchEigenVectorDictionary[i] = finishedX / norm

    def calculateSubNetsImprovedLipschitz(self):
        ms = []
        appliedFlatten = False
        previousX = None
        for i in range(len(self.indexMap) - self.upUntilPenultimate):
            x = self.batchEigenVectorDictionary[i]
            if previousX is not None:
                if self.indexMap[i] > self.flattenIndex and not appliedFlatten:
                    previousX = previousX.flatten(1)
                    appliedFlatten = True
                x = torch.vstack([previousX, x])
            x = self.applyLayerForward(self.layers[self.indexMap[i]], x)
            previousX = x[self.forwardContinuingIndices[i], :]

            remainingXs = x[self.forwardFinishedIndices[i][::-1], :]

            norm = torch.linalg.norm(remainingXs.flatten(1), 2, 1)
            tempCal = []
            multiplier = 1
            for j in range(norm.shape[0]):
                if i == 0:
                    tempCal.append(norm[j])
                elif j < norm.shape[0] - 1:
                    multiplier *= 0.5
                    tempCal.append(norm[j] * ms[i - 1 - j] * multiplier)
                else:
                    tempCal.append(norm[j] * multiplier)
            ms.append(sum(tempCal))
        return ms

    def calculateNetworkLipschitz(self, selfPairDefaultValue=0, naive=False):
        if naive:
            return super().calculateNetworkLipschitz(selfPairDefaultValue)
        self.applyPowerIteration()
        ms = self.calculateSubNetsImprovedLipschitz()
        if not self.perClassLipschitz and not self.pairwiseLipschitz:
            return ms[-1]
        elif self.pairwiseLipschitz and self.numberOfClasses > self.pairwiseClassLimit:
            return self.calculateLargeClassLipschitzConstant(ms[-1], selfPairDefaultValue)
        # handle the last layer
        x = self.batchEigenVectorDictionary[-1]
        appliedFlatten = False
        multiplier = 1
        tempCal = []
        if self.perClassLipschitz:
            numberOfFinishedRowsPerIteration = self.numberOfClasses
        elif self.pairwiseLipschitz:
            numberOfFinishedRowsPerIteration = self.numberOfPairs
        for counter, i in enumerate(range(len(self.indexMap) - 1, -1, -1)):
            if self.indexMap[i] < self.flattenIndex and not appliedFlatten:
                x = x.reshape(x.shape[0], *self.preFlattenShape)
                appliedFlatten = True
            x = self.applyLayerBackward(self.layers[self.indexMap[i]], x)
            norm = torch.linalg.norm(x.flatten(1), 2, 1, keepdim=True)
            # x = x[numberOfFinishedRowsPerIteration:, :]
            if i > 0:
                multiplier *= 0.5
                tempCal.append(norm * ms[-1 - counter] * multiplier)
            else:
                tempCal.append(norm * multiplier)
        tempCal = sum(tempCal)
        if self.perClassLipschitz:
            return tempCal
        finalLipschitz = [[] for _ in range(self.numberOfClasses)]
        count = 0
        for k in range(self.numberOfClasses):
            for j in range(self.numberOfClasses):
                if k == j:
                    finalLipschitz[k].append(torch.tensor(selfPairDefaultValue).to(self.device))
                elif j < k:
                    finalLipschitz[k].append(finalLipschitz[j][k])
                else:
                    finalLipschitz[k].append(tempCal[count])
                    count += 1
            finalLipschitz[k] = torch.hstack(finalLipschitz[k])

        return torch.vstack(finalLipschitz)

    def miniBatchStep(self):  # required to run after every loss.backward()
        super().miniBatchStep()
        for entry in self.batchEigenVectorDictionary:
            entry.detach_()

