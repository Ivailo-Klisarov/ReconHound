from reconhound import WORD_PATTERN, load_first_names
from bs4 import BeautifulSoup
import requests
url = 'https://08-armonia-excursions-project.vercel.app/'
resp = requests.get(url)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, 'html.parser')
text_blocks = [t.strip() for t in soup.stripped_strings if len(t.strip()) <= 120]
first_names = load_first_names()
for block in text_blocks[:50]:
    words = [match.group(0) for match in WORD_PATTERN.finditer(block)]
    allowed = [w for w in words if w.lower() in first_names]
    print(repr(block))
    print('words:', words)
    print('allowed first words:', allowed)
    print('---')

