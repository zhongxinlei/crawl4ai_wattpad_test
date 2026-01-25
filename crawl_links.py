import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig, LXMLWebScrapingStrategy
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

from time import sleep
import random
import requests
from bs4 import BeautifulSoup
import os
import re

link_url = "https://www.wattpad.com/story/21067756-enhancement"
novel_name = "enhancement"
output_path = f"link_folder/{novel_name}_output.jsonl"
jsonl_path = f"link_folder/{novel_name}_output.jsonl"
raw_novel_path = "raw_novel/"


def fetch_and_save_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    try:
        # 发起 HTTP 请求获取网页内容
        random_sleep_time = random.uniform(1, 2)
        # sleep(random_sleep_time)
        print(f"start to fetch: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        html_content = response.text

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # 查找第一个 <pre> 标签
        parents = soup.find_all(class_=lambda c: c and 'page' in c and 'highlighter' in c)
        # p_tags = soup.select('.page.highlighter p')
        all_p_contents = []  # 用于存储所有 <p> 标签的内容
        content = ''
        for parent in parents:
            all_p_contents.extend(parent.find_all('p'))
        for p_tag in all_p_contents:
            # 提取 <p> 标签的纯文本内容
            text = p_tag.get_text(strip=True)
            if text:  # 确保内容非空
                content = content + '\n\n' + text

        # 如果有内容，则写入到文件中
        if content:
           return content
        else:
            print(f"No content found for {url}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None


async def extract_crypto_prices():
    # 1. Define a simple extraction schema
    schema = {
        "name": "chapter links",
        "baseSelector": "div._01L-d ul[aria-label='story-parts'] > li",    # Repeated elements
        "fields": [
            {
                "name": "chapter_name",
                "selector": "div.wpYp-",
                "type": "text"
            },
            {
                "name": "chapter_link",
                "selector": "a._6qJpE",
                "type": "attribute",
                "attribute": "href"
            }
        ]
    }

    # 2. Create the extraction strategy
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    # 3. Set up your crawler config (if needed)
    config = CrawlerRunConfig(
        # e.g., pass js_code or wait_for if the page is dynamic
        # wait_for="css:.crypto-row:nth-child(20)"
        cache_mode = CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
    )

    async with AsyncWebCrawler(verbose=True) as crawler:
        # 4. Run the crawl and extraction
        result = await crawler.arun(
            url=link_url,

            config=config
        )

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        # 5. Parse the extracted JSON
        data = json.loads(result.extracted_content)
        print(f"Extracted {len(data)} chapters")
        print(json.dumps(data[0], indent=2) if data else "No data found")
        with open(output_path, "w", encoding="utf-8") as f:
            if isinstance(data, list):
                # 如果 data 是列表，遍历每个对象
                for item in data:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + "\n")
            elif isinstance(data, dict):
                # 如果 data 是字典，直接写入单行
                json_line = json.dumps(data, ensure_ascii=False)
                f.write(json_line + "\n")
            else:
                raise ValueError("Unsupported data type. Expected list or dict.")


async def crawl_novel(chapter_name, url):
    """异步爬取小说内容"""
    print(f"\n🔍 开始爬取: {chapter_name} - {url}")

    try:
        schema = {
            "name": "chapter links",
            "baseSelector": "div.page.highlighter p",  # Repeated elements
            "fields": [
                {
                    "name": "chapter_content",
                    # "selector": "p",
                    "type": "text"
                },
                {
                    "name": "chapter_id",
                    # "selector": "p",
                    "type": "attribute",
                    "attribute": "data-p-id"
                }
            ]
        }

        # 2. Create the extraction strategy
        extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)
        xml_extraction_strategy = LXMLWebScrapingStrategy(schema)

        # Configure browser settings
        browser_config = BrowserConfig(
            headless=True
        )

        # Configure crawler settings
        crawler_run_config = CrawlerRunConfig(
            scan_full_page=True,
            scroll_delay=2.2,
            wait_for_images=True,
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
            page_timeout=180000,
            # wait_for=".recommended-stories-view"
        )

        # 3. Set up your crawler config (if needed)
        config = CrawlerRunConfig(
            # e.g., pass js_code or wait_for if the page is dynamic
            # wait_for="css:.crypto-row:nth-child(20)"
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
            wait_for=".recommended-stories-view",
            page_timeout=180000
        )
        async with AsyncWebCrawler(verbose=True) as crawler:
            # 4. Run the crawl and extraction
            result = await crawler.arun(
                url=url,
                config=crawler_run_config
            )

            if not result.success:
                print("Crawl failed:", result.error_message)
                return None

            # 5. Parse the extracted JSON
            data = json.loads(result.extracted_content)
            print(f"Extracted {len(data)} chapters")
            print(json.dumps(data[0], indent=2) if data else "No data found")
            if len(data) < 50 and chapter_name != 'Chapter 200':
                print(f'data length less than 50! need double check: {chapter_name}')
            # else:
            content = ''
            for p_tag in data:
                text = p_tag.get('chapter_content')
                if text:  # 确保内容非空
                    if content == '':
                        content = text
                    else:
                        content = content + '\n\n' + text
            # 写入内容到文件
            output_file_path = os.path.join(raw_novel_path, clean_chapter_name(chapter_name))
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                outfile.write(content + '\n\n')  # 写入内容并添加空行分段
                print(f"Content saved to {output_file_path}")

            return None
    except Exception as e:
        print(f"❌ 爬取失败: {str(e)}")
        return None


def clean_chapter_name(s: str):
    if ':' in s:
        return s.split(':', 1)[0]
    else:
        return s

def extract_chapters(jsonl_path):
    """
    从JSONL文件提取章节信息
    参数：
        jsonl_path: JSONL文件路径
    返回：
        包含(chapter_name, chapter_link)元组的列表
    """
    results = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            # 去除首尾空白字符
            stripped_line = line.strip()

            # 跳过空行和注释行
            if not stripped_line or stripped_line.startswith('//'):
                continue

            try:
                # 解析JSON
                data = json.loads(stripped_line)

                # 提取目标字段
                name = data.get('chapter_name')
                link = data.get('chapter_link')

                # 验证字段存在性
                if not name or not link:
                    print(f"警告：第{line_number}行缺少必要字段")
                    continue

                results.append((name, link))

            except json.JSONDecodeError:
                print(f"错误：第{line_number}行JSON解析失败")
            except Exception as e:
                print(f"意外错误（第{line_number}行）: {str(e)}")

    return results

if __name__ == "__main__":
    # asyncio.run(extract_crypto_prices())

    chapter_links = extract_chapters(jsonl_path)
    for chapter_link in chapter_links:
        print(chapter_link[0] + '->' + chapter_link[1])
        asyncio.run(crawl_novel(chapter_link[0],chapter_link[1]))

        # fetch_and_save_content('https://www.wattpad.com/1310315593-dungeon-diver-stealing-a-monster%27s-power-chapter-1')
