#! /bin/bash
set -x
echo Enter the name of the device used for project names:
read deviceName
maxEpoch=300
architecture="8C2F"
numberOfPowerIterations=5
mkdir scriptResults
fileName="accuracies${architecture}_2.txt"
fileDirectory="scriptResults/${fileName}"
#rm $fileDirectory
read experimentNumber
#
# original: alpha 5, r0 0.2 lambda 15
defaultLambda=150
defaultAlpha=12
defaultR0=0.2

lambdaValues=(100 130 170 200)
r0Values=(0.15 0.3)

alphaValues=(5 5 10 10 15 15)
alphaLambdaValues=(110 140 150 180 170 200)
if [ $experimentNumber -le 3 ]; then
  python train.py  --dataset tiny-imagenet --architecture $architecture --batchSize 256 --maxEpoch 300\
   --initialLearningRate 1e-4 --learningRateDecayEpoch 100 --pairwiseLipschitz --numberOfPowerIterations\
    $numberOfPowerIterations --datasetAugmentationDegree 20 --datasetAugmentationTranslation 0.2 --robustStartingEpoch\
     10 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha $defaultAlpha --radiusMaximizationMaximumPenalizingRadius $defaultR0\
       --radiusMaximizationInitialLambda ${lambdaValues[$experimentNumber]} --smallestLearningRate 1e-7 --device cuda:0\
        --projectName $deviceName --saveAccuracies --accuracyFileName $fileName
elif [ $experimentNumber -le 5 ]; then
  python train.py  --dataset tiny-imagenet --architecture $architecture --batchSize 256 --maxEpoch 300\
   --initialLearningRate 1e-4 --learningRateDecayEpoch 100 --pairwiseLipschitz --numberOfPowerIterations\
    $numberOfPowerIterations --datasetAugmentationDegree 20 --datasetAugmentationTranslation 0.2\
     --robustStartingEpoch 10 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha $defaultAlpha\
       --radiusMaximizationMaximumPenalizingRadius ${r0Values[$(expr ${experimentNumber} - 4)]}\
        --radiusMaximizationInitialLambda $defaultLambda --smallestLearningRate 1e-7 --device cuda:0 --projectName\
         $deviceName --saveAccuracies --accuracyFileName $fileName
else
  python train.py  --dataset tiny-imagenet --architecture $architecture --batchSize 256 --maxEpoch 300\
   --initialLearningRate 1e-4 --learningRateDecayEpoch 100 --pairwiseLipschitz --numberOfPowerIterations\
    $numberOfPowerIterations --datasetAugmentationDegree 20 --datasetAugmentationTranslation 0.2\
     --robustStartingEpoch 10 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType softMax\
      --radiusMaximizationAlpha ${alphaValues[$(expr ${experimentNumber} - 6)]}\
       --radiusMaximizationMaximumPenalizingRadius $defaultR0\
        --radiusMaximizationInitialLambda ${alphaLambdaValues[$(expr ${experimentNumber} - 6)]}\
         --smallestLearningRate 1e-7 --device cuda:0 --projectName $deviceName --saveAccuracies\
          --accuracyFileName $fileName
fi
