import os, sys

from crawl4ai import LLMConfig

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import asyncio
import time
import json
import re
from typing import Dict
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)

web_pages = [
"https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
"https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
"https://docs.chaicode.com/youtube/chai-aur-c/functions/",
"https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
"https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
"https://docs.chaicode.com/youtube/chai-aur-c/loops/",
"https://docs.chaicode.com/youtube/chai-aur-c/operators/",
"https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
"https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
"https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
"https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
"https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/",
"https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
"https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
"https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
"https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
"https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
"https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
"https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
"https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
"https://docs.chaicode.com/youtube/chai-aur-django/models/",
"https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
"https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
"https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
"https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
"https://docs.chaicode.com/youtube/chai-aur-git/branches/",
"https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
"https://docs.chaicode.com/youtube/chai-aur-git/github/",
"https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
"https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
"https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
"https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
"https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
"https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
"https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
"https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
"https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
"https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
"https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
"https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
"https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
"https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
"https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
"https://docs.chaicode.com/youtube/getting-started/"
]



# Basic Example - Simple Crawl
async def simple_crawl():
    web_pages_content = {}
    browser_config = BrowserConfig(headless=True)
    crawler_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    for idx, web_page in enumerate(web_pages):
        print(f"Crawling {idx+1} of {len(web_pages)}: {web_page}")
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=web_page, config=crawler_config
            )
            web_pages_content[web_page] = result.markdown
    with open("web_pages_content.json", "w") as f:
        json.dump(web_pages_content, f, indent=4)
    print("Web pages content saved to web_pages_content.json")
    return True

if __name__ == "__main__":
    asyncio.run(simple_crawl())