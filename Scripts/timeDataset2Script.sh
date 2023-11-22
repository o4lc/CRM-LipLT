#! /bin/bash
set -x
echo Enter the name of the device used for project names:
read deviceName
maxEpoch=20
timeFileName="timeDataset4.txt"
mkdir scriptResults
timeFileDirectory="scriptResults/${timeFileName}"
rm $timeFileDirectory
datasets=("MNIST" "CIFAR10" "tiny-imagenet")
architectures=("6C2FMNIST" "6C2F" "6C2FTinyImagenet")
batchSizes=(512 512 256)

# liplt
for i in {2..2}; do
  python train.py  --dataset ${datasets[$i]} --architecture ${architectures[$i]} --batchSize ${batchSizes[$i]}\
   --maxEpoch $maxEpoch --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --pairwiseLipschitz\
    --numberOfPowerIterations 5 --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1\
     --robustStartingEpoch -1 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha 5 --radiusMaximizationMaximumPenalizingRadius 0.2 --radiusMaximizationInitialLambda 15\
       --smallestLearningRate 1e-6 --device cuda:0 --projectName $deviceName --timeEvents --timeFileName $timeFileName
done
# naive
for i in {2..2}; do
  python train.py  --dataset ${datasets[$i]} --architecture ${architectures[$i]} --batchSize ${batchSizes[$i]}\
   --maxEpoch $maxEpoch --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --pairwiseLipschitz\
    --numberOfPowerIterations 5 --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1\
     --robustStartingEpoch -1 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha 5 --radiusMaximizationMaximumPenalizingRadius 0.2 --radiusMaximizationInitialLambda 15\
       --smallestLearningRate 1e-6 --device cuda:0 --projectName $deviceName --timeEvents --modelLipschitzType naive\
        --timeFileName $timeFileName
done

# standard
for i in {2..2}; do
  python train.py  --dataset ${datasets[$i]} --architecture ${architectures[$i]} --batchSize ${batchSizes[$i]}\
   --maxEpoch $maxEpoch --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --pairwiseLipschitz\
    --numberOfPowerIterations 5 --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1\
     --robustStartingEpoch -1 --criterionType standard --smallestLearningRate 1e-6 --device cuda:0\
      --projectName $deviceName --timeEvents --modelLipschitzType naive --timeFileName $timeFileName
done