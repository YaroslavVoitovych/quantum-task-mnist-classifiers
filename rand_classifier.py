import random
import numpy as np

from digit_classifier_interface import DigitClassificationInterface


class RandomMnistClassifier(DigitClassificationInterface):
    def __init__(self, algorithm_name):
        super().__init__(algorithm_name)

    @classmethod
    def prepare_input_data(cls, input_image):
        crop_size = (10, 10)
        width, height = input_image.size
        left = (width - crop_size[0]) / 2
        top = (height - crop_size[1]) / 2
        right = (width + crop_size[0]) / 2
        bottom = (height + crop_size[1]) / 2
        cropped_image = input_image.crop((left, top, right, bottom))
        return np.array(cropped_image)

    def train(self, dataset, model_path):
        raise NotImplementedError()

    def evaluate(self, test_dataset):
        raise NotImplementedError()

    def predict(self, x_test) -> int:
        '''
            Get random output
        '''
        x_test = x_test.flatten()
        random.seed(int(np.round(len(x_test[x_test > 128]) / len(x_test), decimals=3) * 100))
        return int(random.uniform(0,10))
