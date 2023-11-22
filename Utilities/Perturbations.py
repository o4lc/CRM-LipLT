import torch
import numpy as np


def createGloroPerturbation(lipschitzConstants, batchSize, correctLabels, eps):
    lipschitzCopy = lipschitzConstants.T.repeat(batchSize, 1)
    if len(lipschitzConstants.shape) == 0:
        lipschitzCopy = lipschitzCopy + lipschitzConstants
    elif len(lipschitzConstants.shape) == 2 and lipschitzConstants.shape[1] == 1:
        lipschitzCopy = lipschitzCopy + lipschitzConstants[correctLabels, :]
    elif len(lipschitzConstants.shape) == 2:
        lipschitzCopy = lipschitzConstants[correctLabels, :]
    else:
        raise ValueError
    perturbation = lipschitzCopy * eps
    return perturbation


def createLmtPerturbation(lipschitzConstants, batchSize, numberOfClasses, correctLabels, eps):
    if len(lipschitzConstants.shape) < 2:
        # this is the case for when the Lipschitz constant is calculated for the whole network
        pairwisePerturbation = torch.sqrt(torch.tensor(2)) * eps * lipschitzConstants \
                   * torch.ones(batchSize, numberOfClasses).to(lipschitzConstants.device)
    elif lipschitzConstants.shape[1] == 1:
        # this is the case for when the Lipschitz constant is calculated per class
        pairwisePerturbation = eps * torch.ones(batchSize, numberOfClasses).to(lipschitzConstants.device) \
                               * lipschitzConstants.T
        pairwisePerturbation[range(batchSize), :] += pairwisePerturbation[range(batchSize), correctLabels].unsqueeze(1)
    else:
        # This is the case for pairwise Lipschitz constants
        pairwisePerturbation = eps * lipschitzConstants[correctLabels, :]
    pairwisePerturbation[range(batchSize), correctLabels] = 0
    return pairwisePerturbation


def modifyOutputForVerifiedAccuracy(output, perturbation, batchSize, correctLabels,
                                    regularizeOnlyIfCorrectlyClassified=True):
    worstCaseLogits = output + perturbation
    worstCaseLogits[np.arange(0, batchSize), correctLabels] = -torch.inf

    worstCaseLogit = torch.max(worstCaseLogits, 1).values.unsqueeze(1)
    if regularizeOnlyIfCorrectlyClassified:
        incorrectlyClassified = torch.argmax(output, 1) != correctLabels
        worstCaseLogit[incorrectlyClassified] = 0

    modifiedOutput = torch.hstack([output, worstCaseLogit])
    # print(torch.hstack([torch.max(output, 1).values.unsqueeze(1), worstCaseLogit]))
    return modifiedOutput

def smoothLabels(labels, numberOfClasses, smoothingEpsilon):
    smoothedLabels = torch.ones(labels.shape[0], numberOfClasses).to(labels.device) * smoothingEpsilon
    smoothedLabels[range(labels.shape[0]), labels] = 1 - smoothingEpsilon * (numberOfClasses - 1)
    return smoothedLabels