# **Artwork AI Aanalyzer**

# 1. Prerequisites

Before running the code, make sure you have the following installed:

keras==2.12.0 <br>
matplotlib==3.7.2 <br>
numpy==1.24.3 <br>
pandas==2.0.3 <br>
PyYAML==6.0.1 <br>
scikit_learn==1.3.0 <br>
seaborn==0.12.2 <br>
tensorflow==2.12.1 <br>
opencv-python==4.8.0.76 <br>
scikit-learn~=1.3.0 <br>
opencv-python~=4.8.0.76 <br>


# 2. Getting Started

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Set up the configuration parameters in the `config/config.yml` file according to your needs.


# 3. Data Crawling

CrawlDirectory is a Python class designed to facilitate web crawling tasks, such as scraping data from a website. It is flexible and can be configured to perform a sequence of steps for crawling, collecting, and downloading data from a target website.

## 3.1. Configuration

The behavior of the CrawlDirectory class is controlled by a configuration file (crawl_directory_config.yml). In this file, you can specify:

- The base URL of the website to crawl.
- The headless mode to run without opening a visible web browser window
- The active step for crawling (which step to run).
- Maximum retry limits for data retrieval attempts.
- CSS codes for identifying page elements.
- Art categories relevant to your crawling task.

## Usage

Here's how you can use the CrawlDirectory class to perform web crawling:

**1- Initialization:** Create an instance of the CrawlDirectory class by providing the path to your configuration file.
```python
crawler = CrawlDirectory(config_path="path/to/crawl_directory_config.yml")
```

**2- Run the Active Step:** Use the run() method to execute the active step specified in the configuration file. For example:
```python
crawler.run()
```

**3- Resuming Crawling:**

If the crawling process is interrupted, you can easily resume it by following the guidance provided in the log messages. The CrawlDirectory class logs informative messages during the crawling process, including the current state index. To resume crawling after an interruption, you can:

1. **Check the Log Messages:** Examine the log messages generated during the crawling process. The log will include information about the current state index.

2. **Update the Configuration:** Adjust the `current_state_index` parameter in the configuration file to match the appropriate step index found in the log. You can locate the current state index in the log and set it in the configuration file.

By following these steps, you can seamlessly pick up where you left off and ensure a smooth resumption of your web crawling activities, even in the event of interruptions or errors.


## 3.2. Future Recommendations For Crawling

While the provided CrawlDirectory class offers an agile solution for web crawling tasks, it's important to note that it was designed with simplicity and speed in mind, making it suitable for small to medium-sized datasets. For larger and more competitive crawling scenarios, where data volumes are substantial and website complexities are high, the following recommendations can be considered:

These recommendations are geared toward optimizing and scaling your web crawling efforts. They can help you manage larger datasets, store images more efficiently, and handle complex websites with ease. It's important to evaluate your project's specific requirements and constraints when deciding which recommendations to implement.

Ultimately, the choice between an agile custom solution and specialized frameworks depends on factors such as the size of your dataset, website complexity, available resources, and project timeline. These recommendations provide guidance for more extensive and competitive web crawling projects.

### Handling Large Datasets

When dealing with large datasets in web crawling, consider the following recommendations:

#### 1- Image Storage with Minio:

For efficient storage of a large number of images, especially if you plan to crawl a substantial amount of artwork images, you can integrate Minio. Minio is an open-source object storage server that provides scalable and high-performance storage for your image files. By using Minio, you can offload the burden of storing images from your local file system and ensure data durability.

#### 2- Database for Metadata

For storing metadata and structured data associated with crawled content, consider using a relational database management system (RDBMS) like SQLite or PostgreSQL. These databases offer excellent performance and data querying capabilities. You can store information such as artist details, post URLs, image URLs, and category labels in a structured manner.

#### 3- Efficient Crawling with Scrapy and Splash

While the CrawlDirectory class provides a flexible and custom approach to web crawling, for larger-scale and more complex crawling tasks, you may want to explore specialized crawling frameworks. Here are some recommendations:

#### 4- Scrapy

Scrapy is a powerful and widely-used web crawling framework for Python. It offers a range of features, including request handling, automatic retries, and built-in support for handling common web crawling patterns. Scrapy is particularly well-suited for large-scale web scraping tasks.

#### 5- Splash

Splash is a headless browser designed for web scraping and rendering JavaScript-heavy websites. It can be integrated with Scrapy to handle dynamic web pages that require JavaScript execution. Splash provides a convenient way to interact with web pages as if you were using a web browser, making it ideal for scenarios where you need to render JavaScript-driven content.

Using Scrapy and Splash together can enhance your web crawling capabilities and enable you to tackle more complex websites efficiently.

Keep in mind that the choice between the custom approach (such as CrawlDirectory) and specialized frameworks (Scrapy and Splash) depends on the specific requirements of your web crawling project. Consider factors like data volume, and website complexity when making your decision.


# 4. Image labeling

This repository is a collection of code that implements a deep learning model capable of performing multi-label image classification. The model is built using the MobileNetV2 architecture, a popular choice for image classification tasks.
The code allows for the training and testing of the model on a given dataset of images. During training, the model utilizes a two-step process. First, the feature extraction section of the model, which is pretrained, is utilized. This section has been previously trained on a large dataset and is not updated during the training process. This allows the model to benefit from the knowledge learned from a diverse range of images.
The second step of the training process involves re-training the semantic encoding section of the model. This section consists of a fully connected network at the end of the model. During this step, the model is trained specifically on current art images. This allows the model to learn the specific features and characteristics of art images, enhancing its ability to accurately classify them.
To generate predictions, the model uses the sigmoid activation function at the last layer. This function outputs a value between 0 and 1 for each possible label, representing the model's confidence in the presence of that label in the image.
During training, the model utilizes the Binary Cross Entropy loss function to measure the difference between the predicted labels and the ground truth labels. This loss function is commonly used for multi-label classification tasks. Additionally, the accuracy metric is used to evaluate the performance of the model during training, providing a measure of how well the model is able to correctly classify the images.


## 4.1. Training the Model

To train the model, follow these steps:

1. Set up the phase parameter in the `config/config.yml` file for training model.
2. Run the `run_image_labeling.py` script.
3. The script will read the images from the specified dataset directory and preprocess them.
4. The model will be trained using the preprocessed images and the specified parameters.
5. After training, the model will be saved in the specified directory.


### Label Distribution Analysis:

Before training the model, it's important to be aware of the label distribution within your art dataset. An imbalanced dataset, where some labels are much more frequent than others, can lead to biased model performance. To gain insights into the label distribution, you can run the following command:

```bash
python run_label_distribution.py
```

Running this command will generate plots that illustrate the frequency of each label in the dataset. These plots can help you understand the extent of the label imbalance.

<p align="center">
  <a href="https://github.com/hamiGH/artwork-ai-analyzer/blob/main/output/label_distribution.png" target="_blank">
    <img src="https://github.com/hamiGH/artwork-ai-analyzer/blob/main/output/label_distribution.png">
  </a>
</p>

As you can see from the label distribution plots, data may be imbalanced, with certain labels appearing more frequently than others.

### Data Balancing with Custom Data Generator:

To address the label imbalance issue, we have developed a custom data generator (custom_data_generator.py). This generator employs a strategic approach to augment and balance the dataset during training.

Here's how the custom data generator works:

1. **Probability Computation:** We compute a probability for each sample in the dataset based on its label frequency. Labels that occur less frequently will have a higher probability of being selected during batch generation.

2. **Random Batch Generation:** During training, the generator randomly selects a batch of samples based on these computed probabilities. This means that labels with lower frequencies will be oversampled to balance the dataset.

By using this custom data generator, we ensure that the model receives a balanced representation of each label during training, improving its ability to generalize and make accurate predictions.

This approach is particularly effective when dealing with imbalanced datasets, where certain labels may be underrepresented. It helps the model learn from all labels more evenly, resulting in better overall performance.

### Data Augmentation:
In order to enhance the performance and robustness of our models, we have implemented a data augmentation stage that incorporates various transformations. These transformations include rotation, width shift, height shift, zoom range, and horizontal flip.

Rotation: We apply random rotations to the images within a specified range, allowing our models to generalize better to different orientations and angles.

Width Shift and Height Shift: By randomly shifting the width and height of the images, we introduce slight variations in the positioning of objects. This helps our models learn to handle different object placements and improves their ability to adapt to real-world scenarios.

Zoom Range: We apply random zooming to the images, both in and out, to simulate different scales and perspectives. This enables our models to better handle objects at various distances and sizes.

Horizontal Flip: We randomly flip the images horizontally, providing additional training data that captures different viewpoints. This helps our models become more robust to mirror images and different object orientations.

By incorporating these transformations into our data augmentation stage, we effectively increase the diversity and quantity of training data. This allows our models to learn from a wider range of examples, improving their ability to generalize and make accurate predictions in various real-world scenarios.


<p align="center">
  <a href="https://github.com/hamiGH/artwork-ai-analyzer/blob/main/output/Performance1.png" target="_blank">
    <img src="https://github.com/hamiGH/artwork-ai-analyzer/blob/main/output/Performance1.png">
  </a>
</p>


## 4.2. Testing the Model

To test the model, follow these steps:

1. Set up the phase parameter in the `config/config.yml` file for testing model.
2. Run the `run_image_labeling.py` script.
3. The script will load the pre-trained model from the specified directory.
4. The model will be tested on the provided test images.
5. The predictions will be displayed along with the corresponding image.

## 4.3. Recommendations

Following methods are proposed to improve the overal performance of the labeling model:
1. Increasing the size and the number of parameters in the model to enhance learning capacity of the model.
2. Reorganizing the input labels in the dataset for images. It seems that there are some overlapping between decision classes.


# 5. Color Analysis

This code performs color analysis on an image by extracting the dominant colors using K-means clustering and DBSCAN clustering algorithms. It also provides functionality to convert the RGB color values to hexadecimal codes and vice versa.

## 5.1. Usage

To perform color analysis on an image, follow these steps:

1. Run the `run_color_analysis.py` script.
2. Use the `get_top_3_dominant_colors(image_path)` method to extract the top 3 dominant colors from an image specified by the `image_path`.
3. Use the `plot_dominant_colors2(image_path, top_colors)` method to plot the original image and the dominant colors.

| [![Image 1](https://github.com/hamiGH/artwork-ai-analyzer/blob/main/output/dominant_colors_0.png)](https://github.com/hamiGH/artwork-ai-analyzer/blob/main/output/dominant_colors_0.png) | [![Image 2](https://github.com/hamiGH/artwork-ai-analyzer/blob/main/output/dominant_colors_18.png)](https://github.com/hamiGH/artwork-ai-analyzer/blob/main/output/dominant_colors_18.png) |
| ------------------------------------------------- | ------------------------------------------------- |


# 6. Similarity Detection

This code implements a similarity detection algorithm for comparing images based on their semantic vectors and dominant colors (ignored yet). The algorithm utilizes an inference model from image labeling module.

## 6.1. Usage

To perform similarity detection on two images, follow these steps:

1. Run the `run_similarity_detection.py` script.
2. Create two paths for images
3. Smaller output value shows fewer differences and more similarity.

