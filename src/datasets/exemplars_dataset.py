import importlib
from argparse import ArgumentParser

from datasets.memory_dataset import MemoryDataset
from torch.utils.data import DataLoader


class ExemplarsDataset(MemoryDataset):
    """Exemplar storage for approaches with an interface of Dataset"""

    def __init__(self, class_indices, num_exemplars=0, num_exemplars_per_class=0, exemplar_selection='random', base_class_index=0):
        super().__init__({'x': [], 'y': []}, class_indices=class_indices)
        self.max_num_exemplars_per_class = num_exemplars_per_class
        self.max_num_exemplars = num_exemplars
        self.base_class_index = base_class_index
        assert (num_exemplars_per_class == 0) or (num_exemplars == 0), 'Cannot use both limits at once!'
        if exemplar_selection == 'shap':
            selector_cls = getattr(importlib.import_module(name='datasets.shap_exemplars_selection'), 'ShapExemplarsSelector')
        else:
            cls_name = "{}ExemplarsSelector".format(exemplar_selection.capitalize())
            selector_cls = getattr(importlib.import_module(name='datasets.exemplars_selection'), cls_name)
        self.exemplars_selector = selector_cls(self)

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):  # TODO: remove extra parameters
        parser = ArgumentParser("Exemplars Management Parameters")
        _group = parser.add_mutually_exclusive_group()
        _group.add_argument('--num-exemplars', default=0, type=int, required=False,
                            help='Fixed memory, total number of exemplars (default=%(default)s)')
        _group.add_argument('--num-exemplars-per-class', default=0, type=int, required=False,
                            help='Growing memory, number of exemplars per class (default=%(default)s)')
        parser.add_argument('--exemplar-selection', default='random', type=str,
                            choices=['herding', 'random', 'shap'],
                            required=False, help='Exemplar selection strategy (default=%(default)s)')
        return parser.parse_known_args(args)

    def _is_active(self):
        return self.max_num_exemplars_per_class > 0 or self.max_num_exemplars > 0

    def collect_exemplars(self, model, trn_loader, selection_transform, t=None, from_inputs=False):
        if self._is_active():
            selection_loader = trn_loader

            def _contains_dataset(dataset, target):
                if dataset is target:
                    return True
                if hasattr(dataset, 'datasets'):
                    return any(_contains_dataset(child, target) for child in dataset.datasets)
                return False

            if len(self) > 0 and not _contains_dataset(trn_loader.dataset, self):
                selection_loader = DataLoader(
                    trn_loader.dataset + self,
                    batch_size=trn_loader.batch_size,
                    shuffle=False,
                    num_workers=trn_loader.num_workers,
                    pin_memory=trn_loader.pin_memory)

            self.images, self.labels = self.exemplars_selector(
                model, selection_loader, selection_transform, t, from_inputs)
