import torch
import torch.nn as nn
from Losses.CrossEntropy import Criterion
import wandb


class GloRoLoss(Criterion):
    def __init__(self, initialLambda,
                 schedulerConfiguration={'type': 'exponential', 'schedulerCoefficient': 1, 'maximumCoefficient': 1},
                 robustStartingEpoch=0,
                 regularizeOnlyCorrectPredictions=True):
        """
        schedulerConfiguration={'type': 'exponential', 'schedulerCoefficient': 1, 'maximumCoefficient': 1}
        or
        schedulerConfiguration={'type': 'linear', 'maximumEpoch': 100, 'maximumCoefficient': 1}
        or
        schedulerConfiguration = {'type': 'constant'}
        or
        schedulerConfiguration = {'type': 'plateau', 'gamma': 2, 'patience': 10, maximumCoefficient': 1}
        """

        super(GloRoLoss, self).__init__()
        self.initialLambda = initialLambda
        self.lambdaCoefficient = initialLambda
        self.schedulerConfiguration = schedulerConfiguration
        assert initialLambda >= 0

        self.crossEntropy = lambda output, target: nn.functional.cross_entropy(output, target)
        # self.kld = lambda output, target: nn.functional.kl_div(output, target, reduction="batchmean")
        self.kld = nn.KLDivLoss(reduction="batchmean")

        self.log_softmax = lambda x: nn.functional.log_softmax(x, dim=1)
        self.softmax = lambda x: nn.functional.softmax(x, dim=1)

        self.currentEpoch = 0
        self.robustStartingEpoch = robustStartingEpoch
        self.schedulerValueList = []
        self.regularizeOnlyCorrectPredictions = regularizeOnlyCorrectPredictions

    def forward(self, output, y):  # gloroconf
        originalOutput = output[:, :-1]

        # add regularization only if predicted correctly
        _, predicted = originalOutput.max(1)
        batch_correct = predicted.eq(y)
        batch_size = len(batch_correct)
        correct_count = sum(batch_correct).item()
        incorrect_count = batch_size-correct_count

        correctClassificationLoss = self.crossEntropy(originalOutput[batch_correct], y[batch_correct])*correct_count/batch_size
        falseClassificationLoss = self.crossEntropy(originalOutput[~batch_correct], y[~batch_correct])*incorrect_count/batch_size

        robustnessLoss = 0
        if self.currentEpoch >= self.robustStartingEpoch:
            if self.regularizeOnlyCorrectPredictions:
                robustnessBatch = batch_correct
            else:
                robustnessBatch = torch.arange(batch_size)

            firstTerm = self.log_softmax(output[robustnessBatch])
            secondTerm = torch.hstack([self.softmax(originalOutput[robustnessBatch]),
                                       1e-7 * torch.ones((output[robustnessBatch].shape[0], 1)).to(output.device)])
            robustnessLoss = self.kld(firstTerm, secondTerm)
            # robustnessLoss = self.crossEntropy(output[robustnessBatch], secondTerm)

            # print("Classification loss: {}, robustness loss: {}, robustness times lambda: {}"
            #       .format(classificationLoss.item(), robustnessLoss.item(), (robustnessLoss * self.lambdaCoefficient).item()))

        # return falseClassificationLoss + 1 / (
        #         1 + self.lambdaCoefficient) * correctClassificationLoss + self.lambdaCoefficient / (
        #         1 + self.lambdaCoefficient) * robustnessLoss

        wandb.log({"Classification Loss": falseClassificationLoss + correctClassificationLoss,
                   "Robustness Loss": robustnessLoss,
                   "Robustness Times Lambda": robustnessLoss * self.lambdaCoefficient})

        return falseClassificationLoss + correctClassificationLoss + self.lambdaCoefficient * robustnessLoss

    def load_state_dict(self, state_dict, strict: bool = True):
        self.initialLambda = state_dict['initialLambda']
        self.lambdaCoefficient = state_dict['lambdaCoefficient']
        self.schedulerConfiguration = state_dict['schedulerConfiguration']
        self.currentEpoch = state_dict['currentEpoch']
        self.robustStartingEpoch = state_dict['robustStartingEpoch']

    def state_dict(self):
        stateDictionary = {
            'initialLambda': self.initialLambda,
            'lambdaCoefficient': self.lambdaCoefficient,
            'schedulerConfiguration': self.schedulerConfiguration,
            'currentEpoch': self.currentEpoch,
            'robustStartingEpoch': self.robustStartingEpoch
        }
        return stateDictionary

    def schedulerStep(self, value=None):
        self.schedulerValueList.append(value)
        self.currentEpoch += 1
        if self.currentEpoch < self.robustStartingEpoch:
            return
        if self.schedulerConfiguration['type'] == 'exponential':
            self.lambdaCoefficient = self.lambdaCoefficient * self.schedulerConfiguration['schedulerCoefficient']
            if self.lambdaCoefficient > self.schedulerConfiguration['maximumCoefficient']:
                self.lambdaCoefficient = self.schedulerConfiguration['maximumCoefficient']
        elif self.schedulerConfiguration['type'] == 'linear':
            epoch = self.currentEpoch - self.robustStartingEpoch
            if epoch < self.schedulerConfiguration['maximumEpoch']:
                self.lambdaCoefficient = self.initialLambda + (
                        self.schedulerConfiguration['maximumCoefficient'] - self.initialLambda) / self.schedulerConfiguration[
                                             'maximumEpoch'] * epoch
            else:
                self.lambdaCoefficient = self.schedulerConfiguration['maximumCoefficient']
        elif self.schedulerConfiguration['type'] == 'constant':
            return
        else:
            raise NotImplementedError
