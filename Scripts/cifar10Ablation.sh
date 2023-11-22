#! /bin/bash
set -x
echo Enter the name of the device used for project names:
read deviceName
maxEpoch=300
architecture="6C2F"
mkdir scriptResults
fileName="accuraciesCIFAR10Ablation.txt"
fileDirectory="scriptResults/${fileName}"
rm $fileDirectory
#
# original: alpha 5, r0 0.2 lambda 15
defaultLambda=15
defaultAlpha=5
defaultR0=0.2
lambdaValues=(1 5 20 50)
r0Values=(0.15 0.3 0.5)

alphaValues=(1 1 3 3 7 7 10 10)
alphaLambdaValues=(3 10 10 20 20 50 30 100)
for lambdaValue in ${lambdaValues[*]}; do
  python train.py  --dataset CIFAR10 --architecture $architecture --batchSize 512 --maxEpoch $maxEpoch\
   --initialLearningRate 1e-3 --learningRateDecayEpoch 200 --pairwiseLipschitz --numberOfPowerIterations 5\
    --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1 --robustStartingEpoch 10\
     --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha $defaultAlpha --radiusMaximizationMaximumPenalizingRadius $defaultR0\
       --radiusMaximizationInitialLambda $lambdaValue --smallestLearningRate 1e-6 --device cuda:0 --projectName\
        $deviceName --saveAccuracies --accuracyFileName $fileName
done

for r0Value in ${r0Values[*]}; do
  python train.py  --dataset CIFAR10 --architecture $architecture --batchSize 512 --maxEpoch $maxEpoch\
   --initialLearningRate 1e-3 --learningRateDecayEpoch 200 --pairwiseLipschitz --numberOfPowerIterations 5\
    --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1 --robustStartingEpoch 10\
     --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha $defaultAlpha --radiusMaximizationMaximumPenalizingRadius $r0Value\
       --radiusMaximizationInitialLambda $defaultLambda --smallestLearningRate 1e-6 --device cuda:0 --projectName\
        $deviceName --saveAccuracies --accuracyFileName $fileName
done
for i in {0..7}; do
  python train.py  --dataset CIFAR10 --architecture $architecture --batchSize 512 --maxEpoch $maxEpoch\
   --initialLearningRate 1e-3 --learningRateDecayEpoch 200 --pairwiseLipschitz --numberOfPowerIterations 5\
    --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1 --robustStartingEpoch 10\
     --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha ${alphaValues[$i]} --radiusMaximizationMaximumPenalizingRadius $defaultR0\
       --radiusMaximizationInitialLambda ${alphaLambdaValues[$i]} --smallestLearningRate 1e-6 --device cuda:0\
        --projectName $deviceName --saveAccuracies --accuracyFileName $fileName
done

