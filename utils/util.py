import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def reset_weights(m):
  """
    Try resetting model weights to avoid
    weight leakage.
  """
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        #self._data.total[key] += value * n
        #self._data.counts[key] += n
        #self._data.average[key] = self._data.total[key] / self._data.counts[key]
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.total[key] / self._data.counts[key]
    
    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def count_steps(self, key=None):
        """
        Return the number of updates from the last reset
        :return: int
        """
        if key is None:
            key = self._data.index[0]
        return self._data.counts[key]

    def process_metrics(self, metric_ftns, prefix, y_hat, y_true):
        # for met in itertools.chain.from_iterable(metric_ftns):
        for met in metric_ftns:
            report = met(y_hat, y_true)
            if isinstance(report, (list, tuple)):
                for r in report:
                    _n, _v = r
                    self.update(f'{prefix}{_n}', _v)
            else:
                self.update(f'{prefix}{met.get_keys()[0]}', report)
