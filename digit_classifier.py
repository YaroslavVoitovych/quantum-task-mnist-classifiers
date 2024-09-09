from PIL import Image

from cnn_classifier import CNNMnistClassifier
from rf_classifier import RandomForestMnistClassifier
from rand_classifier import RandomMnistClassifier


class DigitClassifier:
    '''
        A class with strategy implementation
    '''
    CLASSIFIERS_DICT = {
        'cnn': CNNMnistClassifier,
        'rf': RandomForestMnistClassifier,
        'rand': RandomMnistClassifier
    }

    def __init__(self, algorithm: str):
        self._classifier_type = DigitClassifier.CLASSIFIERS_DICT.get(algorithm, None)
        self._classifier_model = self._classifier_type(algorithm)
        if self._classifier_model is None:
            raise Exception('Unknown classifier')

    def predict(self, input_data: Image) -> int:
        input_data = self._classifier_type.prepare_input_data(input_data)
        return int(self._classifier_model.predict(input_data))


if __name__ == '__main__':
    # Generated black image
    width, height = 28,28
    black_image = Image.new("L", (width, height), 0)
    print(DigitClassifier('cnn').predict(black_image))
    print(DigitClassifier('rf').predict(black_image))
    print(DigitClassifier('rand').predict(black_image))
