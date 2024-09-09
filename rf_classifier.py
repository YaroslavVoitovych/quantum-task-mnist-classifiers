import numpy as np
from sklearn.ensemble import RandomForestClassifier

from digit_classifier_interface import DigitClassificationInterface


class RandomForestMnistClassifier(DigitClassificationInterface):
    def __init__(self, algorithm_name, n_estimators=50):
        super().__init__(algorithm_name)
        self._classifier = RandomForestClassifier(n_estimators=n_estimators)

    @classmethod
    def prepare_input_data(cls, input_image):
        array_image = np.array(input_image).flatten().reshape(1, -1)
        return array_image

    @staticmethod
    def __is_fitted(model):
        '''
        Checks if the classifier is fitted
        '''
        return hasattr(model, 'estimators_') and model.estimators_ is not None

    def train(self, dataset, model_path):
        raise NotImplementedError()

    def evaluate(self, test_dataset):
        raise NotImplementedError()

    def predict(self, x_test):
        if not self.__is_fitted(self._classifier):
            # Fit RF with dummy input
            self._classifier.fit(X=np.zeros(shape=(1, 784)), y=np.zeros(shape=(1, )))
        return self._classifier.predict(x_test)[0]
