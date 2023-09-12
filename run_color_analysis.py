from src import ColorAnalysis


if __name__ == "__main__":
    image_path = "crawled_data/art_dataset/0.jpg"
    top_colors = ColorAnalysis.get_top_3_dominant_colors(image_path)
    ColorAnalysis.plot_dominant_colors(image_path, top_colors)
