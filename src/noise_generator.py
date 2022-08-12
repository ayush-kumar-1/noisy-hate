"""Label Noise Generator 

These functions are used to generate instance independent label noise (IIN). 
IIN is specified by a noise transition matrix $T$. 
"""
import numpy as np


def flip_labels(
    y: np.ndarray,
    transition_matrix: np.ndarray,
    noise_rate: float,
    num_classes: int,
) -> np.ndarray:
    """Flip labels according to a transition matrix.

    Args:
        transition_matrix: A Tensor of shape (num_classes, num_classes)
            specifying the transition matrix.
        y: An integer Tensor of shape (num_instances) representing the labels in range [0, num_classes-1].
        noise_rate: A float specifying the percentage of labels to be flipped [0,1].
        num_classes: An integer specifying the number of classes.

    Returns:
        A Tensor of shape (num_instances) representing the flipped (noisy) labels.

    Raises:
        ValueError: If the shape of the transition matrix is not (num_classes, num_classes).
        ValueError: If y contains values outside of [0, num_classes-1].
        ValueError: If the noise_rate is not in [0,1].
        ValueError: If the number of classes is not greater than 1.
    """
    pass


def gnerate_symmetric_noise(
    y: np.ndarray, noise_rate: float, num_classes: int
) -> np.ndarray:
    """Flips labels according to a symmetric transition matrix. A symmetric transition
    matrix assumes all off-diagonal transition rates are equal.

    Args:
        y: An integer Tensor of shape (num_instances) representing the labels.
        noise_rate: A float specifying the percentage of labels to be flipped [0,1].
        num_classes: An integer specifying the number of classes.

    Returns:
        A Tensor of shape (num_instances) representing the flipped (noisy) labels.

    Raises:
        ValueError: If y contains values outside of [0, num_classes-1].
        ValueError: If the noise_rate is not in [0,1].
        ValueError: If the number of classes is not greater than 1.
    """
    pass


def generate_pair_noise(
    y: np.ndarray, pairs: np.ndarray, noise_rate: float, num_classes: int
) -> np.ndarray:
    """Flips labels according to a pairwise transition matrix. A pairwise transition
    matrix that a given class will only be only be flipped to a different class with
    probablity noise_rate.

    Args:
        y: An integer Tensor of shape (num_instances) representing the labels.
        noise_rate: A float specifying the percentage of labels to be flipped [0,1].
        num_classes: An integer specifying the number of classes.

    Returns:
        A Tensor of shape (num_instances) representing the flipped (noisy) labels.

    Raises:
        ValueError: If y contains values outside of [0, num_classes-1].
        ValueError: If the noise_rate is not in [0,1].
        ValueError: If the number of classes is not greater than 1.
    """
    pass


def transition_matrix_is_valid(
    transition_matrix: np.ndarray, num_classes: int
) -> bool:
    """Checks if the transition matrix is valid. A transition matrix is valid if
    columns sum to 1.

    Args:
        transition_matrix: A Tensor of shape (num_classes, num_classes)
            specifying the transition matrix.
        num_classes: An integer specifying the number of classes.

    Returns:
        True if the transition matrix is valid, else raises a ValueError.

    Raises:
        ValueError: If the shape of the transition matrix is not (num_classes, num_classes).
        ValueError: If the columns of the transition matrix do not sum to 1.
    """
    pass


def labels_are_valid(y: np.ndarray, num_classes: int) -> bool:
    """Checks if the true labels are valid. Labels are valid if all labels are
    integers belonign to [0, num_classes-1].

    Args:
        y: An integer Tensor of shape (num_instances) representing the labels.
        num_classes: An integer specifying the number of classes.

    Returns:
        True if the labels are valid, else raises a ValueError.

    Raises:
        ValueError: If y contains values outside of [0, num_classes-1].
    """
    pass
