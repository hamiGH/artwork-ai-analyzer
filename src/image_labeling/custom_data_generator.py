import numpy as np
from keras.utils import Sequence


class CustomDataGenerator(Sequence):
    def __init__(self, data, labels, batch_size, sample_probabilities, image_generator):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.sample_probabilities = sample_probabilities
        self.image_generator = image_generator

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        selected_indices = np.random.choice(len(self.data), size=self.batch_size, p=self.sample_probabilities)

        selected_images = self.data[selected_indices]
        selected_labels = self.labels[selected_indices]
        augmented_image = []
        for image_index in range(selected_images.shape[0]):
            augmented_image.append(self.image_generator.random_transform(selected_images[image_index, :, :, :]))

        return np.array(augmented_image), np.array(selected_labels)
