import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

"""
topics = ['Animals', 'Art and design', 'Books', 'Buildings & places',
 'Cartoons', 'Clothing & accessories', 'Comics', 'Computers',
 'Electronics', 'Food and drinks', 'Holiday & events', 'Logos',
 'Miscellaneous', 'Movies', 'Music', 'Mythology', 'Nature', 'People',
 'Plants', 'Religion', 'Space', 'Sports & outdoors', 'Television',
 'Toys', 'Vehicles', 'Video games', 'Weapons']
"""

topics = ["animals/insects", "animals/reptiles", "animals/rodents",
"buildings-and-places/furniture", "buildings-and-places/monuments",
"holiday-and-events/christmas", "music/band-logos", "music/musicians",
"people/body-parts", "people/famous", "people/occupations", "people/sexual",
]

base_url = "http://asciiart.eu/{}"
links = []

for topic in tqdm(topics):
	cleaned_topic = topic.replace('&', 'and').replace(' ', '-').lower()
	url = base_url.format(cleaned_topic)
	response = requests.get(url)
	if response.status_code != 200:
		print("status_code err")
		print(topic)
		continue

	soup = BeautifulSoup(response.content, 'html.parser')
	directory_div = soup.find('div', id='directory')
	if not directory_div: print("directory_div err"); continue

	directory_columns_div = directory_div.find('div', class_='directory-columns')
	if not directory_columns_div: print("directory_columns_div err"); continue

	ul = directory_columns_div.find('ul')
	if not ul: print("ul err"); continue

	for li in ul.find_all('li'):
		a = li.find('a')
		if not a or not a.text: print("a err"); continue
		subtopic = a.text.strip()
		full_url = base_url.format(f"{cleaned_topic}/{a['href'].split('/')[-1]}")
		links.append(full_url)

# Writing the links into a file
with open('layer2_links.txt', 'w') as file:
    for link in tqdm(links):
        file.write(link + '\n')

print(f"Finished writing {len(links)} links to links.txt.")

