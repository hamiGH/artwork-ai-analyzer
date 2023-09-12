import os
import time
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from ..utils import load_params


class InferenceModel:
    def __init__(self, config_dir: str):
        # loading config parameters
        params = load_params(os.path.join(config_dir, "config.yml"))

        self.model = None
        self.save_path = params['saved_models_dir']
        self.training = params['phase'] == "train"
        self.model_name = params['model_name']
        self.input_shape = tuple(params['input_shape'] + [3])
        self.batch_size = params['batch_size']
        self.lr = params['learning_rate']
        self.epochs = params['epochs']

    def create_model(self, classes):
        # defining a CNN using the MobileNetV2 architecture pre-trained on the ImageNet dataset
        pretrained_model = tf.keras.applications.MobileNetV2(
            input_shape=list(self.input_shape),
            include_top=False,
            # weights='imagenet',
            weights="src/mobilenet_v2_weights.h5",
            pooling='avg')

        pretrained_model.trainable = False

        # customizing the inference part of the model
        model = tf.keras.models.Sequential([
            pretrained_model,
            # tf.keras.layers.GlobalAveragePooling2D(),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(len(classes), activation='sigmoid')])

        # configuring the learning process using Adam optimizer
        model.compile(optimizer=Adam(learning_rate=self.lr),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        # model.build()
        print(model.summary())

        return model

    def load_model(self):
        model = load_model(self.save_path + "/" + self.model_name, compile=False)
        model.compile(optimizer=Adam(learning_rate=self.lr),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        return model

    def train_model(self, train_generator, validation_dataset, classes):
        self.model = self.create_model(classes)

        model_path = self.save_path + "/model_" + time.time().__str__().split(".")[0] + ".hdf5"
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)

        history_data = self.model.fit_generator(train_generator, validation_data=(validation_dataset['val_x'],
                                                                                  validation_dataset['val_y']),
                                                epochs=self.epochs, verbose=1, callbacks=[checkpoint])
        predictions = self.model.predict(validation_dataset['test_x'])

        return history_data.history, predictions, validation_dataset['test_y']

    def test_model(self, test_images):
        # testing a pre-saved model
        self.model = self.load_model()
        predictions = self.model.predict(test_images)

        return predictions

    @staticmethod
    def plot_training_performance(history):
        _ = plt.figure(figsize=(20, 7))
        plt.subplot(121)
        plt.plot(history['accuracy'], label='Accuracy')
        plt.plot(history['val_accuracy'], label='Val_Accuracy')
        plt.grid()
        plt.legend()

        plt.subplot(122)
        plt.plot(history['loss'], label='Loss')
        plt.plot(history['val_loss'], label='Val_Loss')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_test_images(self, test_images, classes):
        displayed_images = random.randint(len(test_images), size=6)

        _ = plt.figure(figsize=(15, 15))

        for i, image_index in enumerate(displayed_images):
            plt.subplot(int(f'32{i + 1}'))

            # Scale and reshape image before making predictions
            resized = (test_images[image_index, :, :, :] / 255.0).reshape(-1, self.input_shape[1], self.input_shape[0],
                                                                          self.input_shape[2])
            # Predict results
            predictions = self.model.predict(resized)
            predictions = zip(list(classes), list(predictions[0]))
            predictions = sorted(list(predictions), key=lambda z: z[1], reverse=True)[:5]
            print(predictions)
            # Showing image
            plt.imshow(test_images[image_index, :, :, :])
            plt.title(f'{predictions[0][0]}: {round(predictions[0][1] * 100, 2)}% \n {predictions[1][0]}: {round(predictions[1][1] * 100, 2)}%')

        plt.tight_layout()
        plt.show()
