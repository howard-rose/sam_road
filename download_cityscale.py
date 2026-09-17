import gdown
import zipfile
import os
import pathlib

URL = "https://drive.google.com/uc?export=download&id=1wjkHFebGQHPWo63sXqnjUCPNY_a9VJ5e"
TMP = "cityscale_raw.zip"
DEST = pathlib.Path("cityscale")

print("Downloading CityScale dataset from Google Drive...")
gdown.download(URL, TMP, quiet=False, fuzzy=True)

print("Extracting...")
DEST.mkdir(exist_ok=True)
with zipfile.ZipFile(TMP) as z:
    names = z.namelist()
    print(f"  Archive root entries: {sorted({n.split('/')[0] for n in names})}")
    z.extractall(DEST)

os.remove(TMP)
print(f"Done. Data extracted to {DEST}/")
