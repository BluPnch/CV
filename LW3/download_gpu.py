"""Fetch hash-verified official CUDA wheels for this Windows/Python 3.12 setup."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, parse_qs

import requests

from download_metrics import download


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(value for key, value in attrs if key == 'href')


if __name__ == '__main__':
    root = Path('LW3/.cache/wheels')
    for package, version in [('torch', '2.7.1'), ('torchvision', '0.22.1')]:
        index = f'https://download.pytorch.org/whl/cu118/{package}/'
        response = requests.get(index, timeout=(30, 120))
        response.raise_for_status()
        parser = Links()
        parser.feed(response.text)
        filename = f'{package}-{version}+cu118-cp312-cp312-win_amd64.whl'
        matches = [urljoin(index, link) for link in parser.links
                   if unquote(urlsplit(link).path).endswith('/' + filename)]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one official wheel for {filename}, found {len(matches)}')
        url = matches[0]
        sha256 = parse_qs(urlsplit(url).fragment)['sha256'][0]
        download(url.split('#')[0], root / filename, sha256)
    print('CUDA wheels downloaded and verified', flush=True)
