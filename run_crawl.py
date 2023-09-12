from src.crawl import CrawlDirectory


if __name__ == "__main__":
    crawl_directory = CrawlDirectory("config")
    crawl_directory.run()
