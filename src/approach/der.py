import time
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Dark Experience Replay (DER) approach described in
    Buzzega et al. 'Dark Experience for General Continual Learning: a Strong, Simple Baseline' (NeurIPS, 2020).

    DER extends standard experience replay by storing the model's logits (dark experience)
    alongside the exemplar samples. During replay, it enforces consistency between the
    current model's logits on replayed samples and the stored logits, providing a stronger
    regularization signal than replaying with hard labels alone.
    """

    def __init__(self, model, device, nepochs=250, lr=0.1, lr_min=1e-5, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0002, logger=None, exemplars_dataset=None, der_alpha=0.5, der_beta=0.5,
                 use_der_plus=False, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset, **kwargs)
        self.der_alpha = der_alpha  # weight for logit matching loss
        self.der_beta = der_beta    # weight for label CE loss in DER++ mode
        self.use_der_plus = use_der_plus  # DER++ uses both stored logits and labels

        # Dark experience storage: logits paired with exemplar indices
        self.stored_logits = []   # list of logit tensors, aligned with exemplars
        self.stored_labels = []   # list of label tensors, aligned with exemplars

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        assert (have_exemplars > 0), 'Error: DER needs exemplars.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--der-alpha', default=0.5, type=float, required=False,
                            help='Weight for logit matching loss in DER (default=%(default)s)')
        parser.add_argument('--der-beta', default=0.5, type=float, required=False,
                            help='Weight for label CE loss in DER++ (default=%(default)s)')
        parser.add_argument('--use-der-plus', action='store_true', default=False,
                            help='Use DER++ variant with label loss (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self, t=None):
        """Returns the optimizer"""
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop — DER handles replay manually in train_epoch"""
        # We do NOT merge exemplars into trn_loader here (unlike FT-Mem).
        # Instead, we sample from the dark memory buffer in train_epoch.
        return super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader, val_loader):
        """After training, compute and store logits for exemplars (dark experience)"""
        print(f'DER: Collecting exemplars and computing dark experience for task {t}...')
        clock0 = time.time()

        # First, collect exemplars using the standard mechanism
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        # Then compute and store the logits for all exemplars
        self._store_dark_experience()

        clock1 = time.time()
        print(f' > Dark experience stored ({len(self.stored_logits)} samples), time={clock1 - clock0:.3f}s')

    def _store_dark_experience(self):
        """Compute the current model's logits on all exemplars and store them"""
        if len(self.exemplars_dataset) == 0:
            return

        self.stored_logits = []
        self.stored_labels = []

        dark_loader = torch.utils.data.DataLoader(
            self.exemplars_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0)

        self._model.eval()
        with torch.no_grad():
            for images, targets in dark_loader:
                images, targets = self.format_inputs(images, targets)
                outputs = self.model(images)
                # Store concatenated logits
                cat_logits = torch.cat(outputs, dim=1)
                for i in range(len(targets)):
                    self.stored_logits.append(cat_logits[i].cpu().clone())
                    self.stored_labels.append(targets[i].cpu().clone())
        self._model.train()

    def _sample_dark_memory(self, batch_size):
        """Sample a batch from dark experience memory"""
        if len(self.stored_logits) == 0:
            return None, None, None

        num_samples = min(batch_size, len(self.stored_logits))
        indices = np.random.choice(len(self.stored_logits), num_samples, replace=False)

        # Gather samples, stored logits, and labels
        mem_images = []
        mem_logits = []
        mem_labels = []
        for idx in indices:
            img, lbl = self.exemplars_dataset[idx]
            mem_images.append(torch.tensor(img) if not isinstance(img, torch.Tensor) else img)
            mem_logits.append(self.stored_logits[idx])
            mem_labels.append(self.stored_labels[idx])

        mem_images = torch.stack(mem_images).to(self.device)
        mem_logits = torch.stack(mem_logits).to(self.device)
        mem_labels = torch.stack(mem_labels).to(self.device)

        return mem_images, mem_logits, mem_labels

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch with DER-style replay"""
        self._model.train()

        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)

            # Forward current model on current task data
            outputs = self.model(images)
            cat_outputs = torch.cat(outputs, dim=1)

            # Classification loss on current data
            loss = torch.nn.functional.cross_entropy(cat_outputs, targets)

            # Dark experience replay (only for t > 0 and when we have stored logits)
            if t > 0 and len(self.stored_logits) > 0:
                mem_images, mem_logits, mem_labels = self._sample_dark_memory(len(targets))
                if mem_images is not None:
                    # Forward current model on memory samples
                    mem_outputs = self.model(mem_images)
                    mem_cat_outputs = torch.cat(mem_outputs, dim=1)

                    # Logit matching loss (DER core): MSE between stored and current logits
                    # Only match on the logit dimensions that existed when stored
                    min_dim = min(mem_logits.shape[1], mem_cat_outputs.shape[1])
                    logit_loss = torch.nn.functional.mse_loss(
                        mem_cat_outputs[:, :min_dim],
                        mem_logits[:, :min_dim])
                    loss += self.der_alpha * logit_loss

                    # DER++ additional label loss on memory samples
                    if self.use_der_plus:
                        label_loss = torch.nn.functional.cross_entropy(mem_cat_outputs, mem_labels)
                        loss += self.der_beta * label_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets, features=None):
        """Returns the loss value (used for evaluation only in DER)"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

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
