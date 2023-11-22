# from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
import torch
import warnings


class ExponentialLRWithStopping(ExponentialLR):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay. If gamma is not given, a gamma will be generated to
        make sure the exponential decay ends smoothly in max epoch.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma=None, startingEpoch=0, endingEpoch=torch.inf, leastLearningRate=0,
                 last_epoch=-1, verbose=False, **kwargs):
        self.startingEpoch = startingEpoch
        self.endingEpoch = endingEpoch
        self.leastLearningRate = leastLearningRate
        if gamma is None:
            assert endingEpoch != torch.inf and leastLearningRate != 0
            gamma = (leastLearningRate / optimizer.param_groups[0]['lr']) ** (1 / (endingEpoch - startingEpoch))
        super().__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch < self.startingEpoch:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch < self.endingEpoch:
            return [max(group['lr'] * self.gamma, self.leastLearningRate) for group in self.optimizer.param_groups]
        elif self.last_epoch >= self.endingEpoch:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            raise ValueError

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** (max(min(self.last_epoch, self.endingEpoch) - self.startingEpoch, 0)),
                    self.leastLearningRate) for base_lr in self.base_lrs]


class ExponentialLRWithRestarting(ExponentialLR):
    """
    "restartingEpochs": [100, 200, 300, 350],
                                  "gamma": 0.9,
                                  "leastLearningRate": 1e-6
    """
    def __init__(self, optimizer, gamma=0.9, restartingEpochs=[100], leastLearningRate=0,
                 last_epoch=-1, verbose=False, **kwargs):
        self.restartingEpochs = restartingEpochs
        self.restartingEpochs.append(torch.inf)
        self.leastLearningRate = leastLearningRate
        self.currentBaseEpoch = 0
        super().__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        for i in range(1, len(self.restartingEpochs)):
            if self.restartingEpochs[i - 1] <= self.last_epoch < self.restartingEpochs[i]:
                self.currentBaseEpoch = self.restartingEpochs[i - 1]
                break
        if self.last_epoch == 0 or self.last_epoch in self.restartingEpochs:
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [max(group['lr'] * self.gamma, self.leastLearningRate) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch - self.currentBaseEpoch), self.leastLearningRate)
                for base_lr in self.base_lrs]


def createScheduler(schedulerConfiguration, optimizer):
    if schedulerConfiguration['type'] == "exponentialWithStopping":
        scheduler = ExponentialLRWithStopping(optimizer, **schedulerConfiguration)
    elif schedulerConfiguration['type'] == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, schedulerConfiguration['gamma'])
    elif schedulerConfiguration['type'] == "exponentialWithRestarting":
        scheduler = ExponentialLRWithRestarting(optimizer, **schedulerConfiguration)
    elif schedulerConfiguration['type'] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                         factor=schedulerConfiguration['gamma'],
                                                         patience=2, threshold=0.5,
                                                         threshold_mode='abs', cooldown=0,
                                                         min_lr=schedulerConfiguration['leastLearningRate'],
                                                         eps=1e-08, verbose=True)
    else:
        raise NotImplementedError
    return scheduler


def getSchedulerConfiguration(schedulerName, args, **kwargs):
    if schedulerName == "exponentialWithStopping":
        maxEpoch = args.maxEpoch
        schedulerConfiguration = {"type": "exponentialWithStopping",
                                  "startingEpoch": args.learningRateDecayEpoch,
                                  "endingEpoch": maxEpoch,
                                  "leastLearningRate": args.smallestLearningRate
                                  }
    elif schedulerName == "exponentialWithRestarting":
        schedulerConfiguration = {"type": "exponentialWithRestarting",
                                  # "restartingEpochs": [maxEpoch // 2, maxEpoch // 4, maxEpoch // 8],
                                  "restartingEpochs": [100, 200, 300, 350],
                                  "gamma": 0.9,
                                  "leastLearningRate": args.smallestLearningRate
                                  }
    else:
        raise ValueError
    return schedulerConfiguration