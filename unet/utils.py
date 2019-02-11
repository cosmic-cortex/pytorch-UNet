import os

import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value/normalize for key, value in self.results.items()}
