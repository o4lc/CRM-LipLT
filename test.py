import torch
from train import *
import matplotlib.pyplot as plt


def test(checkpoint, device, drawEpsilonFigures=False):
    dataset = checkpoint['dataset']
    eps = checkpoint['test_eps']

    criterionConfiguration = checkpoint['criterionConfiguration']
    inputPerturbationNorm = checkpoint['inputPerturbationNorm']
    lipschitzNorm = checkpoint['lipschitzNorm']

    networkConfiguration = checkpoint['networkConfiguration']
    net = createNetwork(networkConfiguration, device)
    net.load_state_dict(checkpoint['networkStateDictionary'])
    net.to(device)
    net.eval()

    numberOfClasses = net.numberOfClasses
    with torch.no_grad():
        print("Model Accuracy:")
        modelAccuracy = calculateModelAccuracy(net, dataset, device)
        print("Verified Model Accuracy:")
        verifiedAccuracy = calculateVerifiedAccuracy(net, dataset, eps, criterionConfiguration['type'], numberOfClasses,
                                                     device)
    if inputPerturbationNorm != lipschitzNorm:
        print("Norm of input perturbation is not the same as the lipschitz norm.\n"
              " Have you taken this into consideration?")
        time.sleep(1)
        print("If you say so ...")
    print("Model PGD Accuracy:")
    try:
        pgdAccuracy = calculatePgdAccuracy(net, dataset, eps, inputPerturbationNorm, device)
    except Exception as e:
        print("PGD Accuracy could not be calculated.")
        print(e)
        pgdAccuracy = None

    plt.plot(checkpoint['validationAccuracies'])
    plt.title("Validation accuracies in training")

    identifier = checkpoint['dataset'] + '_' + networkConfiguration["modelType"] + '_' + \
                 criterionConfiguration["type"] + '_' + checkpoint['identifier']
    plotSaveLocation = "Plots/"
    if not os.path.exists(plotSaveLocation):
        os.makedirs(plotSaveLocation)
    plt.savefig(plotSaveLocation + identifier + '_validation.png')

    lipschitzConstants = calculateLipschitzConstants(checkpoint, device)
    drawConfig = ['naive', 'liplt']

    if drawEpsilonFigures:
        draw_eps_figure(checkpoint, device, drawConfig )

    return modelAccuracy, verifiedAccuracy, pgdAccuracy, lipschitzConstants

def extractNetworkWeights(net):
    weights = []
    layerIterator = net
    if hasattr(net, "linear"):
        layerIterator = net.linear
    for layer in layerIterator:
        if type(layer) == torch.nn.modules.linear.Linear or type(layer) == torch.nn.modules.Conv2d:
            weights.append(layer.weight)
    return weights
def draw_eps_figure(checkpoint, device, modelTypeList, plotSaveLocation="Plots/"):
    if not os.path.exists(plotSaveLocation):
        os.makedirs(plotSaveLocation)
    dataset = checkpoint['dataset']
    criterionConfiguration = checkpoint['criterionConfiguration']
    inputPerturbationNorm = checkpoint['inputPerturbationNorm']
    networkConfiguration = checkpoint['networkConfiguration']
    net = createNetwork(networkConfiguration, device)
    for w in reversed(extractNetworkWeights(net)):
        numberOfClasses = w.shape[0]
        break
    identifier = checkpoint['dataset'] + '_' + networkConfiguration["modelType"] + '_' + \
                 criterionConfiguration["type"] + '_' + checkpoint['identifier']

    if checkpoint['inputPerturbationNorm'] != checkpoint['lipschitzNorm']:
        print("Norm of input perturbation is not the same as the lipschitz norm.\n"
              " Have you taken this into consideration?")
        time.sleep(1)
        print("If you say so ...")

    if inputPerturbationNorm <= 2:
        epsilonRange = np.linspace(0.01, 3, 10)
    else:
        epsilonRange = np.linspace(0.01, .5, 10)
    originalModelType = networkConfiguration['modelType']
    fvra = plt.figure(figsize=(10, 6), dpi=100)
    fpgd = plt.figure(figsize=(10, 6), dpi=100)
    plottedPgd = False
    for modelType in modelTypeList:
        networkConfiguration['modelType'] = modelType
        if modelType == 'KW':
            networkConfiguration['modelType'] = originalModelType
        if modelType == originalModelType:
            networkConfiguration["numberOfPowerIterations"] = 1
            loadEigenVectors = True
        else:
            networkConfiguration["numberOfPowerIterations"] = 50
            loadEigenVectors = False
        net = createNetwork(networkConfiguration, device)
        net.load_state_dict(checkpoint['networkStateDictionary'], loadEigenVectors=loadEigenVectors)
        net.to(device)
        net.eval()

        vra_figure = np.zeros(epsilonRange.shape[0])
        pgd_figure = np.zeros(epsilonRange.shape[0])
        for i, eps in enumerate(epsilonRange):
            if modelType == 'KW':
                vra_figure[i] = calculateVerifiedAccuracy(net, dataset, eps, 'KW', numberOfClasses, device)
            else:
                vra_figure[i] = calculateVerifiedAccuracy(net, dataset, eps, criterionConfiguration['type'],
                                                          numberOfClasses, device, verbose=False)
            if not plottedPgd:
                pgd_figure[i] = calculatePgdAccuracy(net, dataset, eps, inputPerturbationNorm, device, verbose=False)

        plt.figure(fvra)
        plt.plot(epsilonRange, vra_figure, label=modelType)
        if not plottedPgd:
            plt.figure(fpgd)
            plt.plot(epsilonRange, pgd_figure, label=modelType)
            plottedPgd = True

    plt.figure(fvra)
    plt.title("Verified Accuracy vs Eps")
    plt.xlabel("Eps")
    plt.ylabel("Verified Accuracy")
    plt.legend()
    plt.savefig(plotSaveLocation + identifier + '_vra.png')
    # plt.show()

    plt.figure(fpgd)
    plt.title("Pgd Accuracy vs Eps")
    plt.xlabel("Eps")
    plt.ylabel("Pgd Accuracy")
    plt.legend()
    plt.savefig(plotSaveLocation + identifier + '_pgd.png')
    # plt.show()
    return


def calculateLipschitzConstants(checkpoint, device):
    lipschitzNorm = checkpoint['lipschitzNorm']
    networkConfiguration = checkpoint['networkConfiguration']
    originalModelType = networkConfiguration['modelType']
    originalPerClassLipschitz = networkConfiguration['perClassLipschitz']
    originalPairwiseLipschitz = networkConfiguration['pairwiseLipschitz']

    print("Calculating Lipschitz constants. Original settings: model type {}, per class lipschitz {}".format(
        originalModelType, originalPerClassLipschitz))
    actualLipschitzConstants = None

    for pairwiseLipschitz in [False, True]:
        for perClassLipschitz in [True, False]:
            if pairwiseLipschitz and perClassLipschitz:
                continue
            for modelType in ["naive", "liplt"]:
                networkConfiguration['modelType'] = modelType
                networkConfiguration['perClassLipschitz'] = perClassLipschitz
                networkConfiguration['pairwiseLipschitz'] = pairwiseLipschitz
                if modelType == originalModelType \
                        and perClassLipschitz == originalPerClassLipschitz \
                        and pairwiseLipschitz == originalPairwiseLipschitz:
                    loadEigenVectors = True
                    networkConfiguration['numberOfPowerIterations'] = 1
                else:
                    loadEigenVectors = False
                    networkConfiguration['numberOfPowerIterations'] = 100
                net = createNetwork(networkConfiguration, device)
                net.to(device)
                net.eval()
                net.load_state_dict(checkpoint['networkStateDictionary'], loadEigenVectors=loadEigenVectors)
                lipschitzConstants = net.calculateNetworkLipschitz()
                if pairwiseLipschitz == originalPairwiseLipschitz and perClassLipschitz == originalPerClassLipschitz:
                    actualLipschitzConstants = lipschitzConstants
                print("Model type {}, per class lipschitz {}, pairwise lipschitz {}, Lipschitz constant(s): (norm {}, average lipschitz: {})".
                      format(modelType, perClassLipschitz, pairwiseLipschitz, lipschitzNorm, torch.mean(lipschitzConstants)))
                print(lipschitzConstants)

    networkConfiguration['modelType'] = originalModelType
    networkConfiguration['perClassLipschitz'] = originalPerClassLipschitz
    networkConfiguration['numberOfPowerIterations'] = 1

    return actualLipschitzConstants


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='Networks/home0')
    parser.add_argument('--overridePairwiseLipschitz', action='store_true', default=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device=torch.device("cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "checkpointDictionary" in checkpoint:
        checkpoint = checkpoint["checkpointDictionary"]
    # checkpoint['networkConfiguration']['numberOfPowerIterations'] = 10000
    if args.overridePairwiseLipschitz:
        checkpoint['networkConfiguration']['overridePairwiseLimit'] = True
        checkpoint['networkConfiguration']['numberOfPowerIterations'] = 100

    test(checkpoint, device, drawEpsilonFigures=True)
