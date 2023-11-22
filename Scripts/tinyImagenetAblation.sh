#! /bin/bash
set -x
echo Enter a name for the project, used for checkpoints and results:
read projectName
maxEpoch=300
architecture="8C2F"
numberOfPowerIterations=5
mkdir scriptResults
fileName="scriptResults/accuracies${architecture}.txt"
rm $fileName
#
# original: alpha 5, r0 0.2 lambda 15
defaultLambda=150
defaultAlpha=12
defaultR0=0.2

lambdaValues=(100 130 170 200)
r0Values=(0.15 0.3)

alphaValues=(5 5 10 10 15 15)
alphaLambdaValues=(110 140 150 180 170 200)
for lambdaValue in ${lambdaValues[*]}; do
  python train.py  --dataset tiny-imagenet --architecture $architecture --batchSize 256\
   --maxEpoch 300 --initialLearningRate 1e-4 --learningRateDecayEpoch 100 --pairwiseLipschitz\
    --numberOfPowerIterations $numberOfPowerIterations --datasetAugmentationDegree 20\
     --datasetAugmentationTranslation 0.2 --robustStartingEpoch 10 --criterionType certifiedRadiusMaximization\
      --radiusMaximizationRobustLossType softMax --radiusMaximizationAlpha $defaultAlpha\
       --radiusMaximizationMaximumPenalizingRadius $defaultR0 --radiusMaximizationInitialLambda $lambdaValue\
        --smallestLearningRate 1e-7 --device cuda:0 --projectName $projectName --saveAccuracies
done

for r0Value in ${r0Values[*]}; do
  python train.py  --dataset tiny-imagenet --architecture $architecture --batchSize 256\
   --maxEpoch 300 --initialLearningRate 1e-4 --learningRateDecayEpoch 100 --pairwiseLipschitz\
    --numberOfPowerIterations $numberOfPowerIterations --datasetAugmentationDegree 20\
     --datasetAugmentationTranslation 0.2 --robustStartingEpoch 10 --criterionType certifiedRadiusMaximization\
      --radiusMaximizationRobustLossType softMax --radiusMaximizationAlpha $defaultAlpha\
       --radiusMaximizationMaximumPenalizingRadius $r0Value --radiusMaximizationInitialLambda $defaultLambda\
        --smallestLearningRate 1e-7 --device cuda:0 --projectName $projectName --saveAccuracies
done
for i in {0..7}; do
  python train.py  --dataset tiny-imagenet --architecture $architecture --batchSize 256\
   --maxEpoch 300 --initialLearningRate 1e-4 --learningRateDecayEpoch 100 --pairwiseLipschitz\
    --numberOfPowerIterations $numberOfPowerIterations --datasetAugmentationDegree 20 --datasetAugmentationTranslation\
     0.2 --robustStartingEpoch 10 --criterionType certifiedRadiusMaximization --radiusMaximizationRobustLossType\
      softMax --radiusMaximizationAlpha ${alphaValues[$i]} --radiusMaximizationMaximumPenalizingRadius $defaultR0\
       --radiusMaximizationInitialLambda ${alphaLambdaValues[$i]}\
        --smallestLearningRate 1e-7 --device cuda:0 --projectName $projectName --saveAccuracies
done

