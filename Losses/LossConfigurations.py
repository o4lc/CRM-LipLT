from Losses import GloRo, CrossEntropy


def getCriterionConfiguration(criterionName, args):
    if criterionName == "standard":
        criterionConfiguration = {"type": "standard"}
    elif criterionName == "GloRo":
        maxEpoch = args.maxEpoch
        criterionConfiguration = \
            {"type": "GloRo",
             "initialLambda": args.gloroInitialLambda,
             "robustStartingEpoch": args.robustStartingEpoch,
             "regularizeOnlyCorrectPredictions": False,
             "schedulerConfiguration": {'type': 'linear',
                                        'maximumEpoch': maxEpoch,  # maximum epoch does not need to equal maxEpoch
                                        'maximumCoefficient': 3}}
        # criterionConfiguration = \
        #     {"type": "GloRo",
        #      "initialLambda": args.gloroInitialLambda,
        #      "robustStartingEpoch": args.robustStartingEpoch,
        #      'regularizeOnlyCorrectPredictions': False,
        #      "schedulerConfiguration": {'type': 'constant'}}

    elif criterionName == "LMT":
        criterionConfiguration = {"type": "LMT",
                                  'robustStartingEpoch': args.robustStartingEpoch,
                                  "initialMu": args.lmtInitialLambda,
                                  "schedulerCoefficient": args.lmtLambdaGamma,
                                  "maximumCoefficient": args.lmtMaximumLambda}
    elif criterionName == "certifiedRadiusMaximization":
        criterionConfiguration = {"type": "certifiedRadiusMaximization",
                                  "robustStartingEpoch": args.robustStartingEpoch,
                                  "initialMu": args.radiusMaximizationInitialLambda,
                                  "schedulerCoefficient": args.radiusMaximizationLambdaGamma,
                                  "maximumCoefficient": args.radiusMaximizationMaximumLambda,
                                  "minimumCoefficient": args.radiusMaximizationMinimumLambda,
                                  "alpha": args.radiusMaximizationAlpha,
                                  "maximumPenalizingRadius": args.radiusMaximizationMaximumPenalizingRadius,
                                  "maximumPenalizingRadiusGamma":
                                      args.radiusMaximizationMaximumPenalizingRadiusGamma,
                                  "maximumPenalizingRadiusMaximumValue":
                                      args.radiusMaximizationMaximumPenalizingRadiusMaximumValue,
                                  "numberOfClasses": args.numberOfClasses,
                                  "crossEntropyType": args.radiusMaximizationCrossEntropyType,
                                  "robustLossType": args.radiusMaximizationRobustLossType}
    else:
        raise ValueError
    return criterionConfiguration


def createCriterion(criterionConfiguration):
    if criterionConfiguration['type'] in ["standard"]:
        criterion = CrossEntropy.CrossEntropy()
    elif criterionConfiguration['type'] == "LMT":
        criterion = CrossEntropy.LMTCrossEntropy(**criterionConfiguration)
    elif criterionConfiguration['type'] == "GloRo":
        criterion = GloRo.GloRoLoss(criterionConfiguration['initialLambda'],
                                    schedulerConfiguration=criterionConfiguration['schedulerConfiguration'],
                                    robustStartingEpoch=criterionConfiguration['robustStartingEpoch'],
                                    regularizeOnlyCorrectPredictions=
                                    criterionConfiguration['regularizeOnlyCorrectPredictions'],)

    elif criterionConfiguration['type'] == "certifiedRadiusMaximization":
        criterion = CrossEntropy.CertifiedRadiusCrossEntropy(**criterionConfiguration)
    else:
        raise NotImplementedError
    return criterion
