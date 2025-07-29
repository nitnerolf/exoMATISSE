#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, subprocess, unicodedata
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from op_tidyup_esofiles import *

def main(root=".", max_workers=None):
    root_path = Path(root)
    # First, rename all .fits and .fits.Z files
    for ext in ("*.fits*", "*.xml", "*.png"):
        for p in root_path.rglob(ext):
            if p.is_file() and is_visible(p):
                safe = sanitize_filename(p.name)
                if safe != p.name:
                    print(f"Renaming {p.name} to {safe}...")
                    p.rename(p.with_name(safe))

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root)