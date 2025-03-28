import os
import urllib.request
import hashlib

def download(url: str, filename: str):
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    dl_path = f"/tmp/semantic-grasping-cache/{url_hash}/{filename}"
    if not os.path.exists(dl_path):
        os.makedirs(os.path.dirname(dl_path), exist_ok=True)
        urllib.request.urlretrieve(url, dl_path)
    return dl_path
