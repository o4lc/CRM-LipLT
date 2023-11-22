#! /bin/bash
set -x
echo Enter a name for the project, used for checkpoints and results:
read projectName
maxEpoch=300
architecture="6C2F"
mkdir scriptResults
fileName="accuraciesCIFARAblation.txt"
fileDirectory="scriptResults/${fileName}"
rm $fileDirectory
#
# original: alpha 5, r0 0.2 lambda 15
defaultLambda=15
defaultAlpha=5
defaultR0=0.2

numberOfPowerIterations=(1 2 5 10)

for numberOfPowerIteration in ${numberOfPowerIterations[*]}; do
  python train.py  --dataset CIFAR10 --architecture $architecture --batchSize 512 --maxEpoch $maxEpoch\
   --initialLearningRate 1e-3 --learningRateDecayEpoch 200 --pairwiseLipschitz --numberOfPowerIterations\
    $numberOfPowerIteration --datasetAugmentationDegree 10 --datasetAugmentationTranslation 0.1\
     --robustStartingEpoch 10 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha $defaultAlpha --radiusMaximizationMaximumPenalizingRadius $defaultR0\
       --radiusMaximizationInitialLambda $defaultLambda --smallestLearningRate 1e-6 --device cuda:0 --projectName\
        $projectName --saveAccuracies --accuracyFileName $fileName
done
