# CRM-LipLT: Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization
Code for paper:

[Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization](https://arxiv.org/abs/2310.00116),
by Mahyar Fazlyab, Taha Entesari, Aniket Roy, and Rama Chellappa. In NeurIPS 2023.

## Dependencies
The code was run on a cuda enabled linux machine with python 3.8.10 and PyTorch 1.11.0+cu113
We were able to reproduce the results just by installing the following packages:
```
pip install numpy matplotlib wandb
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
The code uses wandb for logging. By default, wandb is disabled.
To enable it, you need to configure your own wandb api key. Afterwards, run your codes with the "--wandb" flag.

## Training
We have provided a set of scripts in the "Scripts" folder that were used for the ablation studies.
We note that the scripts are not necessarily the same as the ones used for the paper and they might have been modified.
The scripts should provide an insight into how to run the experiments.

We note that there are slight differences between the notation used in the paper and the code.
Most importantly, the parameter "alpha" (as in "radiusMaximizationAlpha") in the code used in the corresponding loss of CRM maps 
to the hyperparameter "t" in the paper 


We'll provide sample commands for training models on MNIST and CIFAR-10 below:
```bash
python train.py  --dataset MNIST --architecture 4C3F --batchSize 512\
   --maxEpoch 400 --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --pairwiseLipschitz\
    --numberOfPowerIterations 10 --datasetAugmentationDegree 0. --datasetAugmentationTranslation 0.\
     --robustStartingEpoch 10 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha 5 --radiusMaximizationMaximumPenalizingRadius 2.2 --radiusMaximizationInitialLambda 30\
       --smallestLearningRate 1e-6 --device cuda:0 --projectName home
```

```bash
python train.py --dataset CIFAR10 --architecture 6C2F --batchSize 512 --maxEpoch 400\
 --initialLearningRate 1e-3 --learningRateDecayEpoch 200 --pairwiseLipschitz\
  --numberOfPowerIterations 5 --datasetAugmentationDegree 10\
   --datasetAugmentationTranslation 0.1 --robustStartingEpoch 10\
    --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
     --radiusMaximizationAlpha 5 --radiusMaximizationMaximumPenalizingRadius 0.2\
      --radiusMaximizationInitialLambda 15 --smallestLearningRate 1e-6 --device cuda:0\
       --projectName home
```

The above scripts train models on MNIST and CIFAR-10, respectively,
using CRM using LipLT as the pairwise Lipschitz calculation algorithm.
To use the standard Lipschitz calculation algorithm, provide the arguments "--modelLipschitzType naive".
The default value for this is "liplt".

Also, to perform per class Lipschitz calculation (rather than pairwise), remove the "--pairwiseLipschitz" flag
and add the "--perClassLipschitz" flag.
To perform per model Lipschitz calculation, remove both flags.

## Lipschitz Calculation
The current implementation of the code only accepts feed forward models consisting of
linear and convolutional layers without any skip connections.
If you have such a model and want to calculate its Lipschitz constant using LipLT, you can use the'
"sequentialConversion.ipynb" notebook to load your model into our object and calculate its Lipschitz constant. 