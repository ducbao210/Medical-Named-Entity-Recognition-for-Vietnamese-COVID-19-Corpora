"""Download only CoNLL files from VinAIResearch/PhoNER_COVID19."""

import json
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE = "https://api.github.com/repos/VinAIResearch/PhoNER_COVID19/contents/data/word?ref=main"
DATA_DIR = Path(__file__).resolve().parent / "data"


def download_conll_files():
    request = Request(SOURCE, headers={"User-Agent": "phoner-covid19-downloader"})
    with urlopen(request, timeout=30) as response:
        entries = json.load(response)

    files = [entry for entry in entries if entry["type"] == "file" and entry["name"].endswith(".conll")]
    if not files:
        raise RuntimeError("The source directory contains no .conll files")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for entry in files:
        request = Request(entry["download_url"], headers={"User-Agent": "phoner-covid19-downloader"})
        with urlopen(request, timeout=30) as response:
            (DATA_DIR / entry["name"]).write_bytes(response.read())

    for path in DATA_DIR.iterdir():
        if path.is_file() and path.suffix != ".conll":
            path.unlink()

    return sorted(path.name for path in DATA_DIR.glob("*.conll"))


if __name__ == "__main__":
    print("Downloaded:", ", ".join(download_conll_files()))
