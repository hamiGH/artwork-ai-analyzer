import os
import ast
import time
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common import NoSuchElementException, TimeoutException, ElementNotInteractableException, \
    WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.chrome.service import Service as ChromeService
from ..utils import load_params


# Define a class for the crawling program
class CrawlDirectory:

    # Initialize the class with the configuration path
    def __init__(self, config_path: str):
        # Load parameters from configuration file
        params = load_params(os.path.join(config_path, "crawl_directory_config.yml"))

        # Base url of the site
        self.base_url = params['base_url']

        # Initialize crawling steps and active step to run
        self.active_step = params['active_step']
        self.crawling_steps = params['crawling_steps']

        # Data to resume crawling
        self.current_state_index = params['current_state_index']

        # Maximum retry limit for data retrieval attempts.
        self.max_retries = params['max_retries']

        # All CSS codes for page elements required for crawling
        self.css_codes = params['css_codes']

        # All art categories
        self.art_categories = params['art_categories']

        # Initialize driver
        chrome_options = webdriver.ChromeOptions()
        if params['headless_mode']:
            chrome_options.add_argument('--headless')  # Run Chrome in headless mode
            chrome_options.add_argument('--disable-gpu')  # Disable GPU acceleration (recommended for headless)

        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

    # Retrieve artist elements
    def get_artist_elements(self):

        retry_count = 0
        while retry_count < self.max_retries:

            try:

                # Wait for an element to be visible before interacting with it
                wait = WebDriverWait(self.driver, 20)  # Adjust the timeout as needed
                _ = wait.until(expected_conditions.visibility_of_element_located((By.CSS_SELECTOR,
                                                                                  self.css_codes['artist_element'])))

                # Find the artist element to ensure the page is loaded
                artist_elements = self.driver.find_elements(By.CSS_SELECTOR, self.css_codes['artist_element'])
                if artist_elements:
                    return artist_elements

                time.sleep(1)

            except TimeoutException or NoSuchElementException:
                retry_count += 1
                print("The page has not been loaded yet.")
                time.sleep(1)

        print("Maximum retry attempts for the page have been reached.")
        return None

    # Navigate to the next page
    def go_to_next_page(self):

        next_button_is_enabled = False
        retry_count = 0

        while retry_count < self.max_retries:
            try:

                next_button = self.driver.find_element(By.CSS_SELECTOR, self.css_codes['next_button'])
                next_button_is_enabled = (
                        self.driver.find_element(By.CSS_SELECTOR, self.css_codes['next_button_ability'])
                        .get_attribute("aria-disabled") == 'false')

                if next_button_is_enabled:
                    # Click the "Next" button to move to the next page
                    self.driver.execute_script("arguments[0].scrollIntoView();", next_button)
                    # A short delay before clicking
                    time.sleep(1)
                    next_button.click()
                    time.sleep(1)

            except NoSuchElementException as _:
                # Current page does not have the next button
                next_button_is_enabled = False
                return next_button_is_enabled

            except ElementNotInteractableException as ex:
                retry_count += 1
                if retry_count < self.max_retries:
                    print("The Next button is not inter-actable due to an issue with page scrolling.")
                else:
                    print("Max retry attempts for clicking Next button reached due to an issue with page scrolling.")
                    raise ex

            time.sleep(1)

        return next_button_is_enabled

    def crawl_artists(self):

        os.makedirs("crawled_data", exist_ok=True)

        artists = {}
        if self.current_state_index != 0 and os.path.exists("crawled_data/artists.csv"):
            artists_data = pd.read_csv("crawled_data/artists.csv")
            artists_data['category_labels'] = artists_data['category_labels'].apply(ast.literal_eval)
            artists = artists_data.set_index('username')['category_labels'].to_dict()
        else:
            self.current_state_index = 0

        try:
            for category in self.art_categories[self.current_state_index:]:
                print(f"Crawling artists with the category expertise: {category}")
                search_url = self.base_url + '/search/?category=' + category.replace(' ', '-')

                self.driver.get(search_url)
                time.sleep(1)

                next_button_is_enabled = True
                while next_button_is_enabled:  # Loop until the last page

                    artists_elements = self.get_artist_elements()
                    if not artists_elements:
                        print("Category \'{category}\' doesn't have any artist.".format(category=category))

                    for element in artists_elements:
                        href = element.get_attribute('href')
                        print(href)
                        artist_username = href.split("/")[-2]
                        if artist_username in artists:
                            artists[artist_username].append(category)
                        else:
                            artists[artist_username] = [category]

                    next_button_is_enabled = self.go_to_next_page()

                # saving artist and their expertises
                (pd.DataFrame(list(artists.items()), columns=['username', 'category_labels']).
                 to_csv("crawled_data/artists.csv"))
                self.current_state_index += 1

        except Exception as ex:
            print(ex)
            # saving artist and their expertises
            (pd.DataFrame(list(artists.items()), columns=['username', 'category_labels'])
             .to_csv("crawled_data/artists.csv"))
            self.__resume_guide()

    def get_post_elements(self):
        retry_count = 0
        while retry_count < self.max_retries:

            try:

                # Wait for an element to be visible before interacting with it
                wait = WebDriverWait(self.driver, 5)  # Adjust the timeout as needed
                _ = wait.until(expected_conditions.visibility_of_element_located((By.CSS_SELECTOR,
                                                                                  self.css_codes['post_element'])))

                # Find the post element to ensure the page is loaded
                post_elements = self.driver.find_elements(By.CSS_SELECTOR, self.css_codes['post_element'])
                if post_elements:
                    return post_elements

                time.sleep(1)

            except TimeoutException or NoSuchElementException:
                retry_count += 1
                print("The page has not been loaded yet.")
                time.sleep(1)

        print("Maximum retry attempts for the page have been reached.")
        return None

    def crawl_posts(self):

        # load the usernames of artists for crawling their posts
        if os.path.exists("crawled_data/artists.csv"):
            artists_data = pd.read_csv("crawled_data/artists.csv")
            artists_data['category_labels'] = artists_data['category_labels'].apply(ast.literal_eval)
        else:
            print("The 'artists.csv' file does not exist in the output directory. "
                  "Please run the 'crawling_artists' step first.")
            return

        # Load data for resume crawling posts
        posts = {}
        if self.current_state_index != 0 and os.path.exists("crawled_data/posts.csv"):
            posts_data = pd.read_csv("crawled_data/posts.csv")
            posts_data['category_labels'] = posts_data['category_labels'].apply(ast.literal_eval)
            posts = posts_data.set_index('post_url')['category_labels'].to_dict()
        else:
            self.current_state_index = 0

        artists_data = artists_data.iloc[self.current_state_index:]
        try:
            for row_index, row in artists_data.iterrows():

                artist_username = row['username']
                category_labels = row['category_labels']
                print("({}) Artist username: {}".format(row_index, artist_username))

                artist_url = self.base_url + "/artist/" + artist_username
                self.driver.get(artist_url)
                time.sleep(1)

                next_button_is_enabled = True
                while next_button_is_enabled:  # Loop until the last page

                    post_items = self.get_post_elements()
                    if not post_items:
                        print("Artist \'{username}\' doesn't have any post.".format(username=artist_username))

                    else:
                        time.sleep(1)
                        for post_item in post_items:
                            href = post_item.get_attribute('href')
                            print(href)
                            posts[href] = category_labels

                    next_button_is_enabled = self.go_to_next_page()

                (pd.DataFrame(list(posts.items()), columns=['post_url', 'category_labels'])
                 .to_csv("crawled_data/posts.csv"))
                self.current_state_index = row_index + 1

        except Exception as ex:
            print(ex)
            # saving urls of posts
            pd.DataFrame(list(posts.items()), columns=['post_url', 'category_labels']).to_csv("crawled_data/posts.csv")
            self.__resume_guide()

    def go_to_post_page(self, post_url):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                self.driver.get(post_url)
                time.sleep(1)
                return
            except WebDriverException as _:
                retry_count += 1
                if retry_count < self.max_retries:
                    print(f"Retrying... (Attempt {retry_count + 1})")
        print("Max retry attempts reached. Skipping this Post URL.")

    def get_image_url(self):
        retry_count = 0
        while retry_count < self.max_retries:

            try:
                # Wait for an element to be visible before interacting with it
                wait = WebDriverWait(self.driver, 20)  # Adjust the timeout as needed
                element = wait.until(
                    expected_conditions.visibility_of_element_located((By.CSS_SELECTOR,
                                                                       self.css_codes['image_element'])))

                # Check if the element contains an image
                if element.find_elements(By.CSS_SELECTOR, "img.main-Image"):
                    # If an image is found, extract the image source
                    image_element = element.find_element(By.CSS_SELECTOR, "img.main-Image")
                    image_src = image_element.get_attribute("src")
                    if ".gif" in image_src.lower():
                        return None
                    return image_src

                elif element.find_elements(By.CSS_SELECTOR, "video.main-Image"):
                    # If a video is found, you can handle it as needed
                    print("Post contains a Video.")
                    return None

                elif element.find_elements(By.CSS_SELECTOR, "button"):
                    # If a button is found, click it
                    button_element = element.find_element(By.CSS_SELECTOR, "button")
                    button_element.click()
                    time.sleep(1)
                    print("Clicked the acknowledge button")

                else:
                    time.sleep(1)
            except TimeoutException or NoSuchElementException:
                retry_count += 1
                if retry_count < self.max_retries:
                    print("artist post is loading")
                else:
                    print("Max retry attempts for image url reached. Skipping this Post URL.")
                    return None
                time.sleep(1)

    def crawl_images(self):

        # load the urls of posts for crawling their images
        if os.path.exists("crawled_data/posts.csv"):
            posts_data = pd.read_csv("crawled_data/posts.csv")
            posts_data['category_labels'] = posts_data['category_labels'].apply(ast.literal_eval)
        else:
            print("The 'posts.csv' file does not exist in the output directory. "
                  "Please run the 'crawling_posts' step first.")
            return

        images = {}
        if self.current_state_index != 0 and os.path.exists("crawled_data/images.csv"):
            images_data = pd.read_csv("crawled_data/images.csv")
            images_data['category_labels'] = images_data['category_labels'].apply(ast.literal_eval)
            images = images_data.set_index('image_url')['category_labels'].to_dict()
        else:
            self.current_state_index = 0

        posts_data = posts_data.iloc[self.current_state_index:]
        try:
            for row_index, row in posts_data.iterrows():
                post_url = row['post_url']
                category_labels = row['category_labels']
                print("({}) Post url: {}".format(row_index, post_url))

                self.go_to_post_page(post_url)

                image_url = self.get_image_url()
                if image_url:
                    images[image_url] = category_labels

                (pd.DataFrame(list(images.items()), columns=['image_url', 'category_labels'])
                 .to_csv("crawled_data/images.csv"))
                self.current_state_index = row_index + 1

        except Exception as ex:
            print(ex)
            # saving url of images
            (pd.DataFrame(list(images.items()), columns=['image_url', 'category_labels'])
             .to_csv("crawled_data/images.csv"))
            self.__resume_guide()

    @staticmethod
    def __extension_from_url(url: str):
        if url.endswith("&w=1200&q=75"):
            return "." + url.rstrip("&w=1200&q=75").split(".")[-1]
        else:
            ".jpg"

    def download_image(self, image_url):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Download the image
                image_response = requests.get(image_url)
                image_response.raise_for_status()  # Check for HTTP errors
                return image_response.content

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while downloading the image: {e}")
                retry_count += 1
                print(f"Retrying... Attempt {retry_count} of {self.max_retries}")
                time.sleep(1)

            except Exception as e:
                print(f"An error occurred: {e}")
                return None  # Exit the loop on unexpected errors
        return None

    def download_images(self):

        # load the urls of images for downloading them
        if os.path.exists("crawled_data/images.csv"):
            images_data = pd.read_csv("images.csv")
            images_data['category_labels'] = images_data['category_labels'].apply(ast.literal_eval)
        else:
            print("The 'images.csv' file does not exist in the output directory. "
                  "Please run the 'crawling_images' step first.")
            return

        os.makedirs("crawled_data/art_dataset", exist_ok=True)

        art_dataset = {}
        if self.current_state_index != 0 and os.path.exists("crawled_data/art_dataset.csv"):
            art_dataset = pd.read_csv("crawled_data/art_dataset.csv")
            art_dataset['category_labels'] = art_dataset['category_labels'].apply(ast.literal_eval)
            art_dataset = art_dataset.set_index('image_name')['category_labels'].to_dict()
        else:
            self.current_state_index = 0

        images_data = images_data.iloc[self.current_state_index:]
        try:
            for row_index, row in images_data.iterrows():
                image_url = row['image_url']
                labels = row['category_labels']
                print(f"({row_index}) Image url: {image_url}")

                image_content = self.download_image(image_url)
                if image_content is None:
                    print(f"This image url is skipped: {image_url}")
                    continue

                image_name = str(row_index) + self.__extension_from_url(image_url)

                # Save the image in the 'images' directory
                with open(os.path.join('crawled_data/art_dataset', image_name), 'wb') as img_file:
                    img_file.write(image_content)

                art_dataset[image_name] = labels
                (pd.DataFrame(list(art_dataset.items()), columns=['image_name', 'category_labels']).
                 to_csv("crawled_data/art_dataset.csv"))
                self.current_state_index = row_index + 1

        except Exception as ex:
            print(ex)
            # saving image info
            (pd.DataFrame(list(art_dataset.items()), columns=['image_name', 'category_labels']).
             to_csv("crawled_data/art_dataset.csv"))
            self.__resume_guide()

    def __resume_guide(self):
        print("The crawling has not finished.")
        print(f"To resume set current state index to |- {self.current_state_index} -| in the config.")

    # Run the crawling program
    def run(self):
        for step in self.crawling_steps:
            # Call the method corresponding to the active step
            if step['name'] == self.active_step:
                if hasattr(self, step['name']):
                    print(step['description'])
                    getattr(self, step['name'])()
