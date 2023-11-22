import torch
import torchvision
import torchvision.transforms as transforms
import os


def createDataLoaders(dataset="MNIST", validationSplit=0.8, batchSize=1024, numberOfWorkers=4, args=None):
    if args is not None:
        dataset = args.dataset
        numberOfWorkers = args.numberOfWorkers
        batchSize = args.batchSize
        validationSplit = args.validationSplit
        degrees = args.datasetAugmentationDegree
        translation = args.datasetAugmentationTranslation
    else:
        degrees = 5
        translation = 0.1
    if dataset == "MNIST":
        trainTransform = transforms.ToTensor()
        trainSet = torchvision.datasets.MNIST('data', train=True, transform=trainTransform, download=True)
        trainSize = int(len(trainSet) * validationSplit)
        trainSet, validationSet = torch.utils.data.random_split(trainSet, [trainSize, len(trainSet) - trainSize])
        testSet = torchvision.datasets.MNIST('data', train=False, transform=trainTransform, download=True)
    elif dataset == "CIFAR10":
        trainTransform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomAffine(degrees=degrees, translate=(translation, translation))
             ])

        testTransform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])
        trainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=trainTransform)
        trainSize = int(len(trainSet) * validationSplit)
        trainSet, validationSet = torch.utils.data.random_split(trainSet, [trainSize, len(trainSet) - trainSize])
        testSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=testTransform)
    elif dataset == "CIFAR100":
        trainTransform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomAffine(degrees=degrees, translate=(translation, translation))
             ])

        testTransform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])
        trainSet = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=trainTransform)
        trainSize = int(len(trainSet) * validationSplit)
        trainSet, validationSet = torch.utils.data.random_split(trainSet, [trainSize, len(trainSet) - trainSize])
        testSet = torchvision.datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=testTransform)
    elif dataset == "tiny-imagenet":

        """
        To download the dataset, use the instructions in the following github repository:
            https://github.com/tjmoon0104/pytorch-tiny-imagenet
        """
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=degrees, translate=(translation, translation))
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }

        data_dir = 'data/tiny-imagenet-200/'
        trainSet = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"])
        validationSet = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), data_transforms["val"])
        testSet = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), data_transforms["test"])
    else:
        raise ValueError
    dataLoaders = {"train": torch.utils.data.DataLoader(trainSet,
                                                        batch_size=batchSize,
                                                        shuffle=True,
                                                        num_workers=numberOfWorkers),
                   "eval": torch.utils.data.DataLoader(validationSet,
                                                       batch_size=batchSize,
                                                       shuffle=True,
                                                       num_workers=numberOfWorkers),
                   "test": torch.utils.data.DataLoader(testSet,
                                                       batch_size=batchSize,
                                                       shuffle=True,
                                                       num_workers=numberOfWorkers)}
    return dataLoaders