import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt


class ColorAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def read_image(image_path):
        # Load the image
        image = cv2.imread(image_path)

        # Convert it from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image if needed
        image = cv2.resize(image, (224, 224))
        return image

    @staticmethod
    def extract_dominant_colors_kmeans(image, num_colors=3):
        # Reshape the image to be a list of pixels
        pixels = image.reshape(-1, 3)

        # Apply K-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)

        # Get the RGB values of the cluster centers
        dominant_colors = kmeans.cluster_centers_

        return dominant_colors.astype(int)

    @staticmethod
    def extract_dominant_colors_dbscan(image, eps=10, min_samples=5):
        # Reshape the image to be a list of pixels
        pixels = image.reshape(-1, 3)

        # Apply DBSCAN clustering to find dominant colors
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(pixels)

        # Get the labels assigned to each pixel
        labels = dbscan.labels_

        # Count the number of unique labels (clusters)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Calculate the dominant colors by averaging the colors in each cluster
        dominant_colors = []
        for cluster_label in range(num_clusters):
            cluster_mask = (labels == cluster_label)
            cluster_pixels = pixels[cluster_mask]
            cluster_color = np.mean(cluster_pixels, axis=0)
            dominant_colors.append(cluster_color)

        return np.array(dominant_colors).astype(int)

    @staticmethod
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])

    @staticmethod
    def convert_to_hexadecimal(dominant_colors):
        return [ColorAnalysis.rgb_to_hex(color) for color in dominant_colors]

    @staticmethod
    def hex_to_rgb(hex_color):
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        # Convert the color to RGB
        rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return rgb_color

    @staticmethod
    def convert_to_rgb(dominant_colors_hex):
        return [ColorAnalysis.hex_to_rgb(color) for color in dominant_colors_hex]

    @staticmethod
    def get_top_3_dominant_colors(image_path: str):
        image = ColorAnalysis.read_image(image_path)
        dominant_colors = ColorAnalysis.extract_dominant_colors_kmeans(image, num_colors=3)
        color_codes = ColorAnalysis.convert_to_hexadecimal(dominant_colors)
        return color_codes[:3]  # Return the top 3 colors

    @staticmethod
    def plot_dominant_colors(image_path, top_colors):
        # Load the image
        image = ColorAnalysis.read_image(image_path)

        # Create a figure and axis
        fig, axes = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [8, 1]})

        # Plot the image
        axes[0].imshow(image, aspect='auto')
        axes[0].axis('off')

        # Plot the dominant colors
        rgb_colors = ColorAnalysis.convert_to_rgb(top_colors)
        rgb_colors = list(np.array(rgb_colors) / 255)

        axes[1].imshow([rgb_colors], aspect='auto')
        axes[1].axis('off')
        axes[1].set_title("Dominant Colors", fontsize=12)

        plt.show()
