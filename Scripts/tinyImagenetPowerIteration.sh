#! /bin/bash
set -x
echo Enter the name of the device used for project names:
read deviceName
maxEpoch=300
architecture="8C2F"
mkdir scriptResults
fileName="accuraciesTinyImagenetAblation.txt"
fileDirectory="scriptResults/${fileName}"
rm $fileDirectory
#
# original: alpha 5, r0 0.2 lambda 15
defaultLambda=150
defaultAlpha=12
defaultR0=0.2

numberOfPowerIterations=(1 2 5 10)
read numberOfPowerIteration
if [$numberOfPowerIteration -eq 0]
then
  for numberOfPowerIteration in ${numberOfPowerIterations[*]}; do
    python train.py  --dataset tiny-imagenet --architecture $architecture --batchSize 256 --maxEpoch 300\
     --initialLearningRate 1e-4 --learningRateDecayEpoch 100 --pairwiseLipschitz --numberOfPowerIterations\
      $numberOfPowerIteration --datasetAugmentationDegree 20 --datasetAugmentationTranslation 0.2\
       --robustStartingEpoch 10 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
        --radiusMaximizationAlpha $defaultAlpha --radiusMaximizationMaximumPenalizingRadius $defaultR0\
         --radiusMaximizationInitialLambda $defaultLambda --smallestLearningRate 1e-7 --device cuda:0\
          --projectName $deviceName --saveAccuracies  --accuracyFileName $fileName
  done
else
  python train.py  --dataset tiny-imagenet --architecture $architecture --batchSize 256 --maxEpoch 300\
   --initialLearningRate 1e-4 --learningRateDecayEpoch 100 --pairwiseLipschitz --numberOfPowerIterations\
    $numberOfPowerIteration --datasetAugmentationDegree 20 --datasetAugmentationTranslation 0.2 --robustStartingEpoch\
     10 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha $defaultAlpha --radiusMaximizationMaximumPenalizingRadius $defaultR0\
       --radiusMaximizationInitialLambda $defaultLambda --smallestLearningRate 1e-7 --device cuda:0 --projectName\
        $deviceName --saveAccuracies  --accuracyFileName $fileName
fi

