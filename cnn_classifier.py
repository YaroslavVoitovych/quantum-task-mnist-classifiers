import numpy as np
import tensorflow as tf
from keras import layers, models

from digit_classifier_interface import DigitClassificationInterface


class CNNMnistClassifier(DigitClassificationInterface):
    '''
        Implements most generic architecture with 3 sequential Conv layers and poolings.
        Ended with dense layer and softmax activation, that reflects one-hot encoding of 10 digits as possible outputs.
        GPU usage is not configured.
    '''
    def __init__(self, algorithm_name: str):
        super().__init__(algorithm_name)
        self._classifier = self.build_model()

    @classmethod
    def prepare_input_data(cls, input_image):
        np_image = np.array(input_image)
        np_image = np_image.reshape((1, 28, 28, 1)).astype('float32') / 255
        tensor = tf.convert_to_tensor(np_image)
        return tensor

    @staticmethod
    def build_model():
        '''
            Using sequential API from Keras, optimizer - Adam, metric - accuracy,
            loss - categorical crossentropy
        '''
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, dataset, model_path):
        raise NotImplementedError()

    def evaluate(self, test_dataset):
        raise NotImplementedError()

    def predict(self, tf_image):
        prediction = self._classifier.predict(tf_image)
        predicted_class = np.argmax(prediction)
        return predicted_class
