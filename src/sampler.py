import torch as tr
import numpy as np

class ImbalancedDatasetSampler(tr.utils.data.sampler.Sampler):
    def __init__(self, dataset, max_imbalance, num_samples):
        self.num_samples = num_samples
        labels = np.array([l for _,_,l in dataset], dtype=int)
        classw = np.array([max_imbalance / (labels == 0).sum(),
                                     1.0 / (labels == 1).sum()])
        self.weights = classw[labels]
        self.weights /= self.weights.sum()

    def __iter__(self):
        sample = np.random.choice(len(self.weights),
                                  size=self.num_samples,
                                  replace=True,
                                  p=self.weights)
        return iter(sample)

    def __len__(self):
        return self.num_samples

