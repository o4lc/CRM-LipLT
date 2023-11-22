#! /bin/bash
set -x
echo Enter a name for the project, used for checkpoints and results:
read projectName
maxEpoch=20
mkdir scriptResults
rm scriptResults/timedEvents.txt
architectures=("6C2F" "7C2F" "8C2FCifar10" "9C2F" "9C3F" "9C4F" "10C4F" "11C4F")
# liplt
for architecture in ${architectures[*]}; do
  python train.py  --dataset CIFAR10 --architecture $architecture --batchSize 512 --maxEpoch $maxEpoch\
   --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --pairwiseLipschitz --numberOfPowerIterations 5\
    --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1 --robustStartingEpoch -1 --criterionType\
     certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax --radiusMaximizationAlpha 5\
      --radiusMaximizationMaximumPenalizingRadius 0.2 --radiusMaximizationInitialLambda 15 --smallestLearningRate 1e-6\
       --device cuda:0 --projectName $projectName --timeEvents
done

# naive
for architecture in ${architectures[*]}; do
  python train.py  --dataset CIFAR10 --architecture $architecture --batchSize 512 --maxEpoch $maxEpoch\
   --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --pairwiseLipschitz --numberOfPowerIterations 5\
    --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1 --robustStartingEpoch -1 --criterionType\
     certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax --radiusMaximizationAlpha 5\
      --radiusMaximizationMaximumPenalizingRadius 0.2 --radiusMaximizationInitialLambda 15 --smallestLearningRate 1e-6\
       --device cuda:0 --projectName $projectName --timeEvents --modelLipschitzType naive
done

# standard training
for architecture in ${architectures[*]}; do
  python train.py  --dataset CIFAR10 --architecture $architecture --batchSize 512 --maxEpoch $maxEpoch\
   --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --datasetAugmentationDegree 10\
    --datasetAugmentationTranslation 0.1 --criterionType standard --smallestLearningRate 1e-6 --device cuda:0\
     --projectName $projectName --timeEvents --modelLipschitzType naive
done