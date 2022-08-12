"""Implements a variety of methods for learning with label 
noise with a task agnostic approach. Implementations of the following 
papers can be found in the following classes. 

Co-Teaching - Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., … Sugiyama, M. (2018). 
Co-Teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels. arXiv.
https://doi.org/10.48550/arXiv.1804.06872
"""
from abc import ABC as AbstractBaseClass
from abc import abstractmethod

class NoisyLearningModel(AbstractBaseClass):
    """Abstract Base Class for models for learning with noisy labels."""

    @abstractmethod
    def __init__(self, x: np.ndarray, y: np.ndarray): 
        """
        Initializes the model with the given data. 

        Args: 
            x: A np.ndarray of shape (num_instances, num_features) representing the features used during training.
            y: A np.ndarray of shape (num_instances) representing the labels [0, num_classes-1] used during training.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """Trains the model.
        
        Args:
            x: A np.ndarray of shape (num_instances, num_features) representing the features.
            y: A np.ndarray of shape (num_instances) representing the labels.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the labels for a given set of features.        
        Args:
            x: A np.ndarray of shape (num_instances, num_features) representing the features.
        
        Returns:
            A np.ndarray of shape (num_instances) representing the predicted labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts the probabilities for a given set of features.        
        Args:
            x: A np.ndarray of shape (num_instances, num_features) representing the features.
        
        Returns:
            A np.ndarray of shape (num_instances, num_classes) representing the predicted probabilities.
        """
        pass

    @abstractmethod
    def evaluate_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluates the loss for a given set of features and labels.
        
        Args:
            x: A np.ndarray of shape (num_instances, num_features) representing the features.
            y: A np.ndarray of shape (num_instances) representing the labels.
        
        Returns:
            A np.ndarray of shape (num_instances) representing the loss.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the model to a file.
        
        Args:
            path: A string representing the path to the file.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Loads the model from a file.
        
        Args:
            path: A string representing the path to the file.
        """
        pass


class CoTeaching(NoisyLearningModel): 
    """Generic Implementation of Co-Teaching.

    Reference: Co-Teaching - Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., … Sugiyama, M. (2018). 
    Co-Teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels. arXiv.
    https://doi.org/10.48550/arXiv.1804.06872
    """