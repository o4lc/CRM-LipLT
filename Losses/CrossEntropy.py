from abc import ABC

import torch
import torch.nn as nn
import wandb


class Criterion(nn.Module):
    def forward(self, output, y):
        raise NotImplementedError

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        pass

    def state_dict(self):
        pass

    def schedulerStep(self):
        pass


class CrossEntropy(Criterion):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.crossEntropy = lambda output, target: nn.functional.cross_entropy(output, target)

    def forward(self, output, y):
        return self.crossEntropy(output, y)

class BoostedCrossEntropy(Criterion):
    def __init__(self, numberOfClasses):
        super(BoostedCrossEntropy, self).__init__()
        self.numberOfClasses = numberOfClasses

    def forward(self, output, y):
        posits = nn.functional.softmax(output, dim=1)
        correctClassPosit = posits[torch.arange(posits.shape[0]), y]
        alternativeBest = torch.max(posits[torch.arange(self.numberOfClasses).repeat(posits.shape[0], 1).to(y.device)
                                           != y.unsqueeze(1)].reshape(posits.shape[0], -1), 1)[0]
        return (-correctClassPosit.log() - (1 - alternativeBest).log()).mean()

class LMTCrossEntropy(CrossEntropy):
    def __init__(self, initialMu, schedulerCoefficient, maximumCoefficient, robustStartingEpoch, **args):
        super(LMTCrossEntropy, self).__init__()
        self.mu = initialMu
        self.maximumCoefficient = maximumCoefficient
        self.schedulerCoefficient = schedulerCoefficient
        self.robustStartingEpoch = robustStartingEpoch
        self.epoch = 0

    def forward(self, output, y, perturbation):
        if self.epoch < self.robustStartingEpoch:
            return self.crossEntropy(output, y)
        else:
            return self.crossEntropy(output + self.mu * perturbation, y)

    def schedulerStep(self):
        self.epoch += 1
        if self.epoch > self.robustStartingEpoch:
            self.mu = min(self.mu * self.schedulerCoefficient, self.maximumCoefficient)

    def state_dict(self):
        return {
            'mu': self.mu,
            'maximumCoefficient': self.maximumCoefficient,
            'schedulerCoefficient': self.schedulerCoefficient,
            'robustStartingEpoch': self.robustStartingEpoch,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        self.mu = state_dict['mu']
        self.maximumCoefficient = state_dict['maximumCoefficient']
        self.schedulerCoefficient = state_dict['schedulerCoefficient']
        self.robustStartingEpoch = state_dict['robustStartingEpoch']
        self.epoch = state_dict['epoch']


class CertifiedRadiusCrossEntropy(CrossEntropy):
    def __init__(self, initialMu, schedulerCoefficient, maximumCoefficient, minimumCoefficient, robustStartingEpoch,
                 alpha, maximumPenalizingRadius, maximumPenalizingRadiusGamma, maximumPenalizingRadiusMaximumValue,
                 crossEntropyType, numberOfClasses, robustLossType,
                 **kwargs):
        super(CertifiedRadiusCrossEntropy, self).__init__()
        self.mu = initialMu
        self.maximumCoefficient = maximumCoefficient
        self.minimumCoefficient = minimumCoefficient
        self.schedulerCoefficient = schedulerCoefficient
        self.alpha = alpha
        self.maximumPenalizingRadius = maximumPenalizingRadius
        self.maximumPenalizingRadiusGamma = maximumPenalizingRadiusGamma
        self.maximumPenalizingRadiusMaximumValue = maximumPenalizingRadiusMaximumValue
        self.epoch = 0
        self.robustStartingEpoch = robustStartingEpoch
        self.crossEntropyType = crossEntropyType
        self.boostedCrossEntropy = BoostedCrossEntropy(numberOfClasses)
        self.robustLossType = robustLossType

    def forward(self, output, y, certifiedRadii, verbose=False):
        if self.crossEntropyType == "boostedCrossEntropy":
            classificationLoss = self.boostedCrossEntropy(output, y)
        else:
            classificationLoss = self.crossEntropy(output, y)

        certifiedRadius = torch.min(certifiedRadii, 1).values
        smallCertifiedRadius = certifiedRadius[certifiedRadius < self.maximumPenalizingRadius]
        robustnessLoss = torch.tensor(0.0, device=output.device)
        if self.robustLossType == "max":
            robustnessLoss = 1 / self.alpha * torch.exp(-self.alpha * smallCertifiedRadius).sum() / output.shape[0]
        elif self.robustLossType == "softMax":
            radiiInNeedOfPenalization = certifiedRadii[certifiedRadius < self.maximumPenalizingRadius, :]
            if radiiInNeedOfPenalization.shape[0] > 0:
                radiiInNeedOfPenalization = radiiInNeedOfPenalization[radiiInNeedOfPenalization != torch.inf].reshape(
                    radiiInNeedOfPenalization.shape[0], -1)
                robustnessLoss = 1 / self.alpha \
                                 * torch.logsumexp(-self.alpha * radiiInNeedOfPenalization, 1).sum() / output.shape[0]

        if verbose:
            print("Classification loss: {}, robustness loss: {}, robustness times mu: {}"
                  .format(classificationLoss.item(), robustnessLoss.item(), (robustnessLoss * self.mu).item()))
        wandb.log({'robustness Loss': robustnessLoss.item(),
                   'classification Loss': classificationLoss.item(),
                   'robustness Times Mu': (robustnessLoss * self.mu).item(),
                   'maximum certified Radius': 0 if certifiedRadius.shape[0] == 0 else certifiedRadius.max().item(),
                   "average Certified Radius": 0 if certifiedRadius.shape[0] == 0 else certifiedRadius.mean().item(),
                   "average Active Certified Radius": smallCertifiedRadius.mean().item(),})
        if self.epoch < self.robustStartingEpoch:
            return classificationLoss
        return classificationLoss + self.mu * robustnessLoss

    def schedulerStep(self):
        self.epoch += 1
        if self.epoch > self.robustStartingEpoch:
            self.mu = max(min(self.mu * self.schedulerCoefficient, self.maximumCoefficient), self.minimumCoefficient)
            self.maximumPenalizingRadius = min(self.maximumPenalizingRadius * self.maximumPenalizingRadiusGamma,
                                               self.maximumPenalizingRadiusMaximumValue)

    def state_dict(self):
        return {
            'mu': self.mu,
            'maximumCoefficient': self.maximumCoefficient,
            'minimumCoefficient': self.minimumCoefficient,
            'schedulerCoefficient': self.schedulerCoefficient,
            'epoch': self.epoch,
            'robustStartingEpoch': self.robustStartingEpoch,
            'maximumPenalizingRadius': self.maximumPenalizingRadius,
            'maximumPenalizingRadiusGamma': self.maximumPenalizingRadiusGamma,
            'maximumPenalizingRadiusMaximumValue': self.maximumPenalizingRadiusMaximumValue,
        }

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        self.mu = state_dict['mu']
        self.maximumCoefficient = state_dict['maximumCoefficient']
        self.minimumCoefficient = state_dict['minimumCoefficient']
        self.schedulerCoefficient = state_dict['schedulerCoefficient']
        self.epoch = state_dict['epoch']
        self.robustStartingEpoch = state_dict['robustStartingEpoch']
        self.maximumPenalizingRadius = state_dict['maximumPenalizingRadius']
        self.maximumPenalizingRadiusGamma = state_dict['maximumPenalizingRadiusGamma']
        self.maximumPenalizingRadiusMaximumValue = state_dict['maximumPenalizingRadiusMaximumValue']


