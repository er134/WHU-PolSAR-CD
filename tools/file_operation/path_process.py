import os
import glob
from pathlib import Path

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', 'npy'

def build_imglist(path):

    # *.txt file with img/vid/dir on each line
    if isinstance(path, str) and Path(path).suffix == '.txt':
        path = Path(path).read_text().rsplit()
    files = []

    for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
        # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
        p = str(Path(p).absolute())
        if '*' in p:
            files.extend(sorted(glob.glob(p, recursive=True)))  # glob
        elif os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
        elif os.path.isfile(p):
            files.append(p)  # files
        else:
            raise FileNotFoundError(f'{p} does not exist')

    return [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
