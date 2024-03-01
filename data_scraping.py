import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs

def get_pre_info_from_div(div):
    """Extracts the pre class suffix and text from a div."""
    pre_tag = div.find('pre')
    if pre_tag:
        pre_class = pre_tag.get('class')
        if pre_class:
            class_suffix = pre_class[0].split('-')[-1]
        else: class_suffix = None
        text = pre_tag.text
        return class_suffix, text
    return None, None

def scrape_site(url, failed: list):
    """Scrapes a given URL for specific divs and returns extracted information."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # find div with class asciiarts
    main_div = soup.find('div', class_='asciiarts')
    if not main_div:
        print(f'Failed to find main div for {url}')
        failed.append(url)
        return []
    divs = main_div.find_all('div', class_='border-header border-top p-3')
    return [get_pre_info_from_div(div) for div in divs]

def process_links(file_path):
    """Reads links from a file and scrapes each site."""
    outputs = []
    failed = []
    with open(file_path, 'r') as file:
        for link in tqdm(file):
            link = link.strip()

            parsed_url = urlparse(link)
            path_parts = parsed_url.path.strip('/').split('/')
            topic = "/".join(path_parts[1:])
            pre_info_list = scrape_site(link, failed)
            
            for pre_suffix, pre_text in pre_info_list:
                if not pre_text: continue
                outputs.append({
                    'topic': topic,
                    'pre_suffix': pre_suffix,
                    'text': pre_text
                })

    return outputs

def main():
    file_path = 'data/ascii_links.txt'
    outputs = process_links(file_path)
    with open('data/ascii_art_data.json', 'w') as file:
        json.dump(outputs, file, indent=4)

if __name__ == '__main__':
    main()