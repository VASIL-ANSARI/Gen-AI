import os, json, hashlib

HASH_FILE = "ingested_hashes.json"

def load_hashes():
    if not os.path.exists(HASH_FILE):
        return set()
    try:
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()

def save_hashes(hashes: set):
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(list(hashes), f)

def compute_hash(text: str) -> str:
    # Use SHA1 for small registry; change to sha256 if desired.
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
