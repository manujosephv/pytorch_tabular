import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


def _make_smooth_weights_for_balanced_classes(y_train, mu=0.15):
    labels_dict = {
        label: count for label, count in zip(np.unique(y_train), np.bincount(y_train))
    }
    total = np.sum(list(labels_dict.values()))
    keys = sorted(labels_dict.keys())
    weight = []
    for i in keys:
        score = np.log(mu * total / float(labels_dict[i]))
        weight.append(score if score > 1 else 1)
    return weight


def get_class_weighted_cross_entropy(y_train, mu=0.15):
    assert y_train.ndim==1, "Utility function only works for binary classification"
    y_train = LabelEncoder().fit_transform(y_train)
    weights = _make_smooth_weights_for_balanced_classes(y_train, mu=0.15)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
    return criterion


def get_balanced_sampler(y_train):
    assert y_train.ndim==1, "Utility function only works for binary classification"
    y_train = LabelEncoder().fit_transform(y_train)
    class_sample_counts = np.bincount(y_train)
    # compute weight for all the samples in the dataset
    # samples_weights contain the probability for each example in dataset to be sampled
    class_weights = 1.0 / torch.Tensor(class_sample_counts)
    train_samples_weight = [class_weights[class_id] for class_id in y_train]
    # now lets initialize samplers
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        train_samples_weight, len(y_train)
    )
    return train_sampler
