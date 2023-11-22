#! /bin/bash
set -x
echo Enter a name for the project, used for checkpoints and results:
read projectName
maxEpoch=10000
timeFileName="timePowerIteration3.txt"
mkdir scriptResults
timeFileDirectory="scriptResults/${timeFileName}"
rm $timeFileDirectory
datasets=("MNIST" "CIFAR10" "tiny-imagenet")
architectures=("4C3F" "6C2F" "8C2F")
batchSizes=(512 512 128)
numberOfPowerIterations=(1 2 5 10)

# liplt
for i in {0..2}; do
  for j in {0..3}; do
    python train.py  --dataset ${datasets[$i]} --architecture ${architectures[$i]} --batchSize ${batchSizes[$i]}\
     --maxEpoch $maxEpoch --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --pairwiseLipschitz\
      --numberOfPowerIterations ${numberOfPowerIterations[$j]} --datasetAugmentationDegree 10\
       --datasetAugmentationTranslation 0.1 --robustStartingEpoch -1 --criterionType certifiedRadiusMaximization\
        --radiusMaximizationRobustLossType softMax --radiusMaximizationAlpha 5\
         --radiusMaximizationMaximumPenalizingRadius 0.2 --radiusMaximizationInitialLambda 15\
          --smallestLearningRate 1e-6 --device cuda:0 --projectName $projectName --timeEvents\
           --timeFileName $timeFileName --pureLipschitzCalculation
  done
done
# naive
for i in {0..2}; do
  for j in {0..3}; do
    python train.py  --dataset ${datasets[$i]} --architecture ${architectures[$i]} --batchSize ${batchSizes[$i]}\
     --maxEpoch $maxEpoch --initialLearningRate 1e-3 --learningRateDecayEpoch 25 --pairwiseLipschitz\
      --numberOfPowerIterations ${numberOfPowerIterations[$j]} --datasetAugmentationDegree 10\
       --datasetAugmentationTranslation 0.1 --robustStartingEpoch -1 --criterionType certifiedRadiusMaximization\
        --radiusMaximizationRobustLossType softMax --radiusMaximizationAlpha 5\
         --radiusMaximizationMaximumPenalizingRadius 0.2 --radiusMaximizationInitialLambda 15\
          --smallestLearningRate 1e-6 --device cuda:0 --projectName $projectName --timeEvents --modelLipschitzType naive\
           --timeFileName $timeFileName --pureLipschitzCalculation
  done
done


