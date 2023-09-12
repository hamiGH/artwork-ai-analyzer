from src import SimilarityDetection
from src import ColorAnalysis

if __name__ == "__main__":
    image_path1 = "../art_dataset/1.png"
    image_path2 = "../art_dataset/2.png"

    image1 = ColorAnalysis.read_image(image_path1)
    image2 = ColorAnalysis.read_image(image_path2)

    similarity_mse = SimilarityDetection.compare_similarity_images(image1, image2)

    print(similarity_mse)