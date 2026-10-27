"""Download UCI Bank Marketing (dataset 222) and extract bank-additional-full.csv."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_DIR = Path(__file__).resolve().parent
TARGET = DATA_DIR / "bank-additional-full.csv"


def main():
    if TARGET.exists():
        print(f"Already present: {TARGET}")
        return
    print(f"Downloading {URL}")
    outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
    name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
    TARGET.write_bytes(inner.read(name))
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
