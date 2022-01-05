"""
Data Processing
--------------
"""

import numpy as np

def normalize_features(
    data: np.ndarray,
    target: int,
    norm: bool,
    norm_values: tuple[float, float],
) -> np.ndarray:
    """Normalize the feature data"""
    # separate features and targets
    targets = data[:, target]
    features = np.delete(data, target, 1)
    # normalize
    if norm == True:
        features_min = np.min(features)
        features_max = np.max(features)
        y = (features - (features_max + features_min) / 2) / (
            features_max - features_min
        )
        features = (
            y * (norm_values[1] - norm_values[0])
            + (norm_values[1] + norm_values[0]) / 2
        )
    # repack data - targets last
    norm_data = np.c_[features, targets]
    return norm_data


def split_xy(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split data into x (features) and y (targets)"""
    y = data[:, -1]
    x = np.delete(data, -1, 1)
    return x, y


def split_data(data: np.ndarray, split: tuple[float, float]) -> tuple[np.ndarray, ...]:
    """Create the train, test and validation datasets"""
    n_rows = data.shape[0]
    np.random.shuffle(data)
    if sum(split) >= 0.99:  # 66/33 method
        train_idx = int(round(n_rows * split[0]))
        train = data[:train_idx, :]
        test = data[train_idx:, :]
        val = train
    else:  # 33/33/33 method
        train_idx = int(round(n_rows * split[0]))
        test_idx = int(round(n_rows * split[1]) + train_idx)
        train = data[:train_idx, :]
        test = data[train_idx:test_idx, :]
        val = data[test_idx:, :]
    # x, y splits
    x_train, y_train = split_xy(train)
    x_test, y_test = split_xy(test)
    x_val, y_val = split_xy(val)

    return x_train, y_train, x_test, y_test, x_val, y_val