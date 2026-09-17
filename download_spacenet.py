import gdown
import zipfile
import os
import pathlib

URL = "https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W"
TMP = "spacenet_raw.zip"
DEST = pathlib.Path("spacenet")

print("Downloading SpaceNet dataset from Google Drive...")
gdown.download(URL, TMP, quiet=False)

print("Extracting...")
DEST.mkdir(exist_ok=True)
with zipfile.ZipFile(TMP) as z:
    names = z.namelist()
    print(f"  Archive root entries: {sorted({n.split('/')[0] for n in names})}")
    z.extractall(DEST)

os.remove(TMP)
print(f"Done. Data extracted to {DEST}/")
print("Expected layout: spacenet/RGB_1.0_meter/")
