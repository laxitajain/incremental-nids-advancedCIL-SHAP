import time
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Elastic Weight Consolidation (EWC) approach described in
    Kirkpatrick et al. 'Overcoming catastrophic forgetting in neural networks' (PNAS, 2017).

    EWC is a regularization-based CIL approach that penalizes changes to parameters
    that are important for previously learned tasks. Importance is measured via the
    diagonal of the Fisher Information Matrix computed on the training data of each task.
    """

    def __init__(self, model, device, nepochs=250, lr=0.1, lr_min=1e-5, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0002, logger=None, exemplars_dataset=None, ewc_lambda=5000,
                 online=False, gamma=1.0, fisher_num_samples=-1, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma  # decay factor for online EWC
        self.fisher_num_samples = fisher_num_samples  # -1 means all samples

        # Storage for Fisher information and old parameters
        self.fisher = {}       # {param_name: Fisher diagonal tensor}
        self.old_params = {}   # {param_name: parameter values after training task t}
        self.older_params = {} # For online EWC running average

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--ewc-lambda', default=5000, type=float, required=False,
                            help='EWC regularization strength (default=%(default)s)')
        parser.add_argument('--online', action='store_true', default=False,
                            help='Use online EWC with running Fisher estimate (default=%(default)s)')
        parser.add_argument('--gamma', default=1.0, type=float, required=False,
                            help='Decay factor for online EWC Fisher accumulation (default=%(default)s)')
        parser.add_argument('--fisher-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for Fisher estimation, -1 for all (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self, t=None):
        """Returns the optimizer"""
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # Add exemplars to train_loader if available (optional memory support)
        if self.exemplars_dataset is not None and len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory)

        # Standard training loop
        return super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader, val_loader):
        """After training, compute Fisher Information Matrix and store parameters"""
        print(f'Computing Fisher Information Matrix for task {t}...')
        clock0 = time.time()

        # Compute Fisher Information on the training data
        fisher = self._compute_fisher(t, trn_loader)

        if self.online and t > 0:
            # Online EWC: accumulate Fisher with decay
            for name in fisher:
                if name in self.fisher:
                    self.fisher[name] = self.gamma * self.fisher[name] + fisher[name]
                else:
                    self.fisher[name] = fisher[name]
        else:
            # Standard EWC: store Fisher for this task (overwrite for simplicity)
            if t == 0:
                self.fisher = fisher
            else:
                # Accumulate Fisher across tasks
                for name in fisher:
                    if name in self.fisher:
                        self.fisher[name] = self.fisher[name] + fisher[name]
                    else:
                        self.fisher[name] = fisher[name]

        # Store the current parameters as the reference point
        self.old_params = {name: param.clone().detach()
                          for name, param in self._model.named_parameters() if param.requires_grad}

        clock1 = time.time()
        print(f' > Fisher computation done, time={clock1 - clock0:.3f}s')

        # Exemplar management (optional)
        if self.exemplars_dataset is not None and self.exemplars_dataset._is_active():
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def _compute_fisher(self, t, trn_loader):
        """Compute the diagonal of the Fisher Information Matrix"""
        fisher = {}
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        self._model.eval()
        num_samples = 0

        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)

            # Forward pass
            outputs = self.model(images)
            # Use the log-softmax of the concatenated outputs
            cat_outputs = torch.cat(outputs, dim=1)
            log_probs = torch.nn.functional.log_softmax(cat_outputs, dim=1)

            # Use the model's predictions (not true labels) for Fisher
            preds = cat_outputs.argmax(dim=1)
            selected_log_probs = log_probs[range(len(preds)), preds]

            for log_prob in selected_log_probs:
                self._model.zero_grad()
                log_prob.backward(retain_graph=True)

                for name, param in self._model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.detach() ** 2

                num_samples += 1

                if self.fisher_num_samples > 0 and num_samples >= self.fisher_num_samples:
                    break

            if self.fisher_num_samples > 0 and num_samples >= self.fisher_num_samples:
                break

        # Normalize
        for name in fisher:
            fisher[name] /= num_samples

        return fisher

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self._model.train()

        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)
            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets, features=None):
        """Returns the loss value: cross-entropy + EWC penalty"""
        # Classification loss
        ce_loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

        # EWC regularization penalty (only for t > 0)
        ewc_loss = 0
        if t > 0 and len(self.fisher) > 0:
            for name, param in self._model.named_parameters():
                if param.requires_grad and name in self.fisher and name in self.old_params:
                    ewc_loss += (self.fisher[name] * (param - self.old_params[name]) ** 2).sum()

        total_loss = ce_loss + (self.ewc_lambda / 2.0) * ewc_loss
        return total_loss

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        outputs_tot, targets_tot, features_tot = [], [], []
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self._model.eval()
            for images, targets in val_loader:
                images, targets = self.format_inputs(images, targets)
                outputs, features = self.model(images, return_features=True)
                outputs_tot.extend(np.concatenate([o.cpu().numpy() for o in outputs], axis=1).tolist())
                features_tot.extend(features.tolist())
                targets_tot.extend(targets.tolist())
                loss = self.criterion(t, outputs, targets)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot
