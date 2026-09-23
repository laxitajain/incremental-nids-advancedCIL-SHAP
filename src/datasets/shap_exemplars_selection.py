"""SHAP-Guided Exemplar Selection

Selects exemplars based on their SHAP profile representativeness — choosing samples
whose SHAP values are closest to the per-class centroid in SHAP-space. This produces
a more "explainably diverse" memory that captures the characteristic feature importance
patterns of each class.
"""

import time
from typing import Iterable

import numpy as np
import torch
from approach.incremental_learning import format_inputs
from datasets.exemplars_dataset import ExemplarsDataset
from networks.network import LLL_Net
from torch.utils.data import DataLoader


class ShapExemplarsSelector:
    """Exemplar selector that uses SHAP values to select representative samples.

    For each class, this selector:
    1. Computes SHAP values for all candidate samples using DeepSHAP
    2. Computes the per-class SHAP centroid (mean SHAP profile)
    3. Selects samples closest to the centroid in SHAP-space (L2 distance)

    This ensures the memory contains samples that are most representative of each
    class's characteristic feature importance pattern, as opposed to random selection
    or feature-space herding which don't consider explainability.
    """

    def __init__(self, exemplars_dataset: ExemplarsDataset):
        self.exemplars_dataset = exemplars_dataset
        self.already_added_classes = ()

    def __call__(self, model: LLL_Net, trn_loader: DataLoader, transform, t=None, from_inputs=False):
        from copy import deepcopy

        tmp_model = deepcopy(model)
        if t is not None:
            tmp_model._modules['heads'] = tmp_model._modules['heads'][:t + 1]
            tmp_model.task_cls = tmp_model.task_cls[:t + 1]
            tmp_model.task_offset = tmp_model.task_offset[:t + 1]

        clock0 = time.time()
        exemplars_per_class = self._exemplars_per_class_num(tmp_model)

        # Change loader to sequential
        sel_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False,
                                num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

        selected_indices = self._select_indices(
            tmp_model, sel_loader, exemplars_per_class, transform, from_inputs)
        self.already_added_classes = set(self._get_labels(sel_loader))

        x, y = zip(*(trn_loader.dataset[idx] for idx in selected_indices))

        clock1 = time.time()
        print('| Selected {:d} SHAP-guided exemplars, time={:5.1f}s'.format(len(x), clock1 - clock0))
        return x, y

    def _exemplars_per_class_num(self, model: LLL_Net):
        if self.exemplars_dataset.max_num_exemplars_per_class:
            return self.exemplars_dataset.max_num_exemplars_per_class

        num_cls = model.task_cls.sum().item()
        num_exemplars = self.exemplars_dataset.max_num_exemplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. Limit of exemplars: {}".format(num_cls, num_exemplars)
        return exemplars_per_class

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader,
                        exemplars_per_class: int, transform, from_inputs=None) -> Iterable:
        """Select exemplars using SHAP-based representativeness."""

        model_device = next(model.parameters()).device

        # Step 1: Extract all inputs and labels
        all_inputs = []
        all_labels = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                all_inputs.append(images if isinstance(images, torch.Tensor) else torch.tensor(images))
                all_labels.extend(targets.numpy() if isinstance(targets, torch.Tensor) else targets)

        all_inputs = torch.cat(all_inputs, dim=0)
        all_labels = np.array(all_labels)

        # Step 2: Try to compute SHAP values, fall back to gradient-based importance if SHAP is unavailable
        try:
            shap_values = self._compute_shap_values(model, all_inputs, model_device)
        except Exception as e:
            print(f'[SHAP Exemplar Selection] SHAP computation failed ({e}), using gradient-based fallback')
            shap_values = self._compute_gradient_importance(model, all_inputs, all_labels, model_device)

        # Step 3: Select exemplars closest to per-class SHAP centroid
        result = []
        for curr_cls in np.unique(all_labels):
            cls_ind = np.where(all_labels == curr_cls)[0]
            assert len(cls_ind) > 0, f"No samples for class {curr_cls}"

            if exemplars_per_class < len(cls_ind):
                # Get SHAP profiles for this class
                cls_shap = shap_values[cls_ind]

                # Compute centroid (mean SHAP profile)
                centroid = cls_shap.mean(axis=0)

                # Compute distances to centroid
                distances = np.linalg.norm(cls_shap - centroid, axis=1)

                # Select the closest samples
                sorted_idx = np.argsort(distances)[:exemplars_per_class]
                result.extend(cls_ind[sorted_idx])
            else:
                print(f'WARNING: Not enough samples for class {curr_cls}: selected ALL.')
                result.extend(list(cls_ind))

        return result

    def _compute_shap_values(self, model, all_inputs, device):
        """Compute DeepSHAP values for all inputs."""
        import shap

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                outputs = self.model(x)
                return torch.cat(outputs, dim=1)

        wrapped = ModelWrapper(model)

        # Use a small background set
        num_background = min(50, len(all_inputs))
        background = all_inputs[:num_background].float().to(device)

        explainer = shap.DeepExplainer(wrapped, background)

        # Compute in batches to manage memory
        batch_size = 128
        all_shap = []
        for i in range(0, len(all_inputs), batch_size):
            batch = all_inputs[i:i + batch_size].float().to(device)
            sv = explainer.shap_values(batch)
            if isinstance(sv, list):
                # Average across classes
                sv = np.mean([np.abs(s) for s in sv], axis=0)
            else:
                sv = np.abs(sv)
            # Flatten spatial dims: (batch, 1, num_pkts, num_fields) -> (batch, num_pkts*num_fields)
            sv = sv.reshape(sv.shape[0], -1)
            all_shap.append(sv)

        return np.concatenate(all_shap, axis=0)

    def _compute_gradient_importance(self, model, all_inputs, all_labels, device):
        """Fallback: compute gradient-based feature importance (faster than SHAP)."""
        print('[SHAP Exemplar Selection] Using gradient-based importance as fallback...')

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                outputs = self.model(x)
                return torch.cat(outputs, dim=1)

        wrapped = ModelWrapper(model)
        wrapped.eval()

        all_importance = []
        batch_size = 128

        for i in range(0, len(all_inputs), batch_size):
            batch = all_inputs[i:i + batch_size].float().to(device)
            batch.requires_grad = True

            labels = torch.tensor(all_labels[i:i + batch_size]).long().to(device)
            outputs = wrapped(batch)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            # Use absolute gradient as importance proxy
            importance = batch.grad.abs().cpu().numpy()
            importance = importance.reshape(importance.shape[0], -1)
            all_importance.append(importance)

            wrapped.zero_grad()

        return np.concatenate(all_importance, axis=0)

    @staticmethod
    def _get_labels(sel_loader):
        if hasattr(sel_loader.dataset, 'labels'):
            return np.asarray(sel_loader.dataset.labels)
        elif hasattr(sel_loader.dataset, 'datasets'):
            labels = []
            for ds in sel_loader.dataset.datasets:
                labels.extend(ds.labels)
            return np.array(labels)
        else:
            raise RuntimeError(f"Unsupported dataset: {sel_loader.dataset.__class__.__name__}")
