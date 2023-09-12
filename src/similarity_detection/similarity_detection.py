import numpy as np
from ..image_labeling import InferenceModel
from ..color_analysis import ColorAnalysis


class SimilarityDetection:
    def __init__(self, config_dir: str):
        model_object = InferenceModel(config_dir=config_dir)
        self.model = model_object.load_model()

    @staticmethod
    def calculate_mse(vector1, vector2):
        return np.mean((vector1 - vector2) ** 2)

    def create_image_semantic_vector(self, image):
        return self.model.predict(image)

    @staticmethod
    def extract_dominant_colors(image):
        return ColorAnalysis.extract_dominant_colors_kmeans(image)

    def compare_similarity_images(self, image1, image2):
        semantic_vector1 = self.create_image_semantic_vector(image1)
        semantic_vector2 = self.create_image_semantic_vector(image2)
        mse1 = self.calculate_mse(semantic_vector1, semantic_vector2)

        top_colors1 = np.array(self.extract_dominant_colors(image1))
        top_colors2 = np.array(self.extract_dominant_colors(image2))

        # we do not use dominant colors for similarity calculations at the moment
        top_colors1 = top_colors1 / 255
        top_colors2 = top_colors2 / 255

        return mse1
