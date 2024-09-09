import logging
from abc import ABC, abstractmethod


class DigitClassificationInterface(ABC):
    '''
        Class for implementing digit classification interface
    '''
    def __init__(self, algorithm_name):
        self._logger = logging.Logger(name=algorithm_name, level=logging.INFO)

    @classmethod
    @abstractmethod
    def prepare_input_data(cls, input_image):
        pass

    @abstractmethod
    def predict(self, input_data):
        """Takes input data and returns predictions."""
        pass

    @abstractmethod
    def train(self, dataset, model_path):
        pass


    @abstractmethod
    def evaluate(self, test_dataset):
        pass


