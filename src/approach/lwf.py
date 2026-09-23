from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning without Forgetting (LwF) approach described in
    Li & Hoiem 'Learning without Forgetting' (TPAMI, 2017).

    LwF uses knowledge distillation from the old model to the new model. Before training
    on new task data, the old model's outputs on the new data are recorded and used as
    soft targets during training. No exemplar memory is required.
    """

    def __init__(self, model, device, nepochs=250, lr=0.1, lr_min=1e-5, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0002, logger=None, exemplars_dataset=None, lwf_lambda=1.0, T=2, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset, **kwargs)
        self.lwf_lambda = lwf_lambda
        self.T = T
        self.model_old = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--lwf-lambda', default=1.0, type=float, required=False,
                            help='Distillation loss weight for LwF (default=%(default)s)')
        parser.add_argument('--T', default=2, type=float, required=False,
                            help='Temperature scaling for distillation (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self, t=None):
        """Returns the optimizer"""
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def post_train_process(self, t, trn_loader, val_loader):
        """After training, save a frozen copy of the current model for distillation"""
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self._model.train()

        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)

            # Forward old model to get distillation targets
            targets_old = None
            if t > 0 and self.model_old is not None:
                with torch.no_grad():
                    targets_old = self.model_old(images)

            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets, targets_old)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clipgrad)
            self.optimizer.step()

    def _distillation_loss(self, outputs_new, outputs_old, t):
        """Compute knowledge distillation loss using temperature-scaled softmax.

        The distillation loss is computed over the old task heads only,
        comparing the current model's outputs to the old model's soft targets.
        """
        # Concatenate old task outputs (heads 0..t-1)
        old_outputs_new = torch.cat(outputs_new[:t], dim=1)
        old_outputs_old = torch.cat(outputs_old[:t], dim=1)

        # Temperature-scaled softmax
        soft_targets = torch.nn.functional.softmax(old_outputs_old / self.T, dim=1)
        log_probs = torch.nn.functional.log_softmax(old_outputs_new / self.T, dim=1)

        # KL divergence (scaled by T^2 as per Hinton et al.)
        kd_loss = torch.nn.functional.kl_div(log_probs, soft_targets, reduction='batchmean') * (self.T ** 2)

        return kd_loss

    def criterion(self, t, outputs, targets, targets_old=None, train=True):
        """Returns the loss value: classification loss + distillation loss"""
        # Classification loss on all task outputs
        ce_loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

        # Distillation loss (only for t > 0)
        dist_loss = 0
        if t > 0 and targets_old is not None:
            dist_loss = self._distillation_loss(outputs, targets_old, t)

        # Weight the losses: lamb controls the balance
        # More old classes → more weight on distillation
        if t > 0:
            lamb = (self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()).to(self.device)
        else:
            lamb = 0

        total_loss = (1.0 - lamb) * ce_loss + self.lwf_lambda * lamb * dist_loss
        return total_loss

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        outputs_tot, targets_tot, features_tot = [], [], []
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self._model.eval()
            for images, targets in val_loader:
                images, targets = self.format_inputs(images, targets)

                # Forward old model for loss computation
                targets_old = None
                if t > 0 and self.model_old is not None:
                    targets_old = self.model_old(images)

                outputs, features = self.model(images, return_features=True)
                outputs_tot.extend(np.concatenate([o.cpu().numpy() for o in outputs], axis=1).tolist())
                features_tot.extend(features.tolist())
                targets_tot.extend(targets.tolist())
                loss = self.criterion(t, outputs, targets, targets_old, train=False)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot
