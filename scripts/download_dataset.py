import hashlib
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "dataset"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TARGET_PER_CATEGORY = 5
SEARCH_LIMIT_PER_CATEGORY = 30
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png'}
COMMONS_API_URL = 'https://commons.wikimedia.org/w/api.php'
USER_AGENT = 'animal-semantic-search-prototype/0.1 (local dataset preparation)'

# ---- throttling & backoff ----
MAX_RETRIES = 5
BASE_BACKOFF = 2.0          # seconds; doubles each retry
SEARCH_DELAY = 10.0          # seconds between API search calls
DOWNLOAD_DELAY = 5.0        # seconds between individual image downloads
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

CATEGORIES = {
    'cat': ['cat animal photograph', 'domestic cat full body', 'cat portrait'],
    'dog': ['dog animal photograph', 'domestic dog full body', 'dog portrait'],
    'horse': ['horse animal photograph', 'horse full body', 'brown horse photograph'],
    'tiger': ['tiger animal photograph', 'bengal tiger photograph', 'tiger stripes photograph'],
}
def safe_html_text(value):
    if not value:
        return None
    text = value.get('value') if isinstance(value, dict) else str(value)
    if text is None:
        return None
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip() or None

def guess_extension_from_url(url):
    suffix = Path(urlparse(url).path.lower()).suffix
    if suffix in ALLOWED_EXTENSIONS:
        return '.jpg' if suffix == '.jpeg' else suffix
    return None


def robust_http_get(url, *, params=None, stream=False, timeout=60):
    """HTTP GET with safe handling of Wikimedia rate limits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                stream=stream,
                timeout=timeout,
                headers={'User-Agent': USER_AGENT},
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get('Retry-After', '600')
                print(
                    f"Rate limited by Wikimedia. "
                    f"Retry after {retry_after} seconds. Stopping this request.",
                    flush=True
                )
                resp.close()
                raise SystemExit("Wikimedia rate limit reached. Stopping the entire download.")

            resp.raise_for_status()
            return resp

        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == MAX_RETRIES:
                raise
            delay = BASE_BACKOFF * (2 ** (attempt - 1))
            print(f"Connection issue. Retrying in {delay} seconds...")
            time.sleep(delay)

    raise RuntimeError("Maximum retries exceeded")



def search_commons_images(query, limit=50):
    params = {
        'action': 'query', 'format': 'json', 'generator': 'search',
        'gsrsearch': query, 'gsrnamespace': 6, 'gsrlimit': limit,
        'prop': 'imageinfo', 'iiprop': 'url|mime|size|sha1|extmetadata',
    }
    response = robust_http_get(COMMONS_API_URL, params=params, timeout=30)
    data = response.json()
    pages = data.get('query', {}).get('pages', {})
    results = []
    for page in pages.values():
        title = page.get('title')
        imageinfo = page.get('imageinfo', [])
        if not title or not imageinfo:
            continue
        info = imageinfo[0]
        url = info.get('url')
        mime = info.get('mime')
        if not url or mime not in ALLOWED_MIME_TYPES:
            continue
        ext = guess_extension_from_url(url)
        if ext not in {'.jpg', '.png'}:
            continue
        extmetadata = info.get('extmetadata', {})
        results.append({
            'title': title,
            'url': url,
            'mime': mime,
            'size': info.get('size'),
            'sha1': info.get('sha1'),
            'extension': ext,
            'license': safe_html_text(extmetadata.get('LicenseShortName')),
            'artist': safe_html_text(extmetadata.get('Artist')),
            'credit': safe_html_text(extmetadata.get('Credit')),
        })
    return results
def is_relevant_animal_image(item, category):
    title = item["title"].lower()

    unwanted_terms = [
        "sculpture", "statue", "carving", "frieze",
        "drawing", "painting", "illustration",
        "toy", "plush", "logo", "cartoon",
        "taxidermy", "decoration"
    ]

    if any(term in title for term in unwanted_terms):
        return False

    return True

def download_file(url, destination):
    with robust_http_get(url, stream=True, timeout=60) as response:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def validate_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.format in {'JPEG', 'PNG'} and img.width >= 128 and img.height >= 128
    except (UnidentifiedImageError, OSError):
        return False

def normalize_image_extension(path, category, index):
    with Image.open(path) as img:
        fmt = img.format
    final_ext = '.jpg' if fmt == 'JPEG' else '.png'
    final_path = path.parent / f'{category}_{index:03d}{final_ext}'
    if path != final_path:
        path.rename(final_path)
    return final_path

def prepare_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, global_hashes = [], set()
    for category, queries in CATEGORIES.items():
        print(f'\n=== {category} ===', flush=True)
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        collected = 0
        candidates, seen_urls = [], set()
        for query in queries:
            print(f'Searching: {query}', flush=True)
            try:
                candidates.extend(search_commons_images(query, limit=SEARCH_LIMIT_PER_CATEGORY))
            except Exception as exc:
                print(f'  Search failed for {query!r}: {exc}', flush=True)
            time.sleep(SEARCH_DELAY)
        unique_candidates = []
        for item in candidates:
            if item['url'] in seen_urls:
                continue

            seen_urls.add(item['url'])

            if is_relevant_animal_image(item, category):
                unique_candidates.append(item)
        for item in unique_candidates:
            if collected >= TARGET_PER_CATEGORY:
                break
            next_index = collected + 1
            temp_path = category_dir / f'__tmp_{category}_{next_index}{item["extension"]}'
            try:
                print(f'  Downloading candidate {next_index}: {item["title"]}', flush=True)
                download_file(item['url'], temp_path)
                if not validate_image(temp_path):
                    print('    skipped invalid/unsupported', flush=True)
                    temp_path.unlink(missing_ok=True)
                    continue
                digest = file_sha256(temp_path)
                if digest in global_hashes:
                    print('    skipped duplicate', flush=True)
                    temp_path.unlink(missing_ok=True)
                    continue
                global_hashes.add(digest)
                final_path = normalize_image_extension(temp_path, category, next_index)
                collected += 1
                item_manifest = {
                    'animal': category,
                    'filename': final_path.name,
                    'relative_path': str(final_path.relative_to(OUTPUT_DIR)),
                    'source': 'Wikimedia Commons',
                    'source_url': item['url'],
                    'source_file_title': item['title'],
                    'mime': item['mime'],
                    'sha256': digest,
                }
                for k in ['license', 'artist', 'credit']:
                    if item.get(k): item_manifest[k] = item[k]
                manifest.append(item_manifest)
            except Exception as exc:
                print(f'    skipped error: {exc}', flush=True)
                temp_path.unlink(missing_ok=True)
                continue
            time.sleep(DOWNLOAD_DELAY)
        print(f'Collected {collected}/{TARGET_PER_CATEGORY} for {category}', flush=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'\nDone: {MANIFEST_PATH}; total images: {len(manifest)}', flush=True)

if __name__ == '__main__':
    prepare_dataset()
