#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, subprocess, unicodedata
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii"))

def is_visible(path: Path):
    return not any(part.startswith('.') for part in path.parts)

def compress_fits_file(f: Path):
    try:
        gz = f.with_suffix(".fits.gz")
        print(f"... {f.name} → {gz.name}")
        with open(f, "rb") as fi, open(gz, "wb") as fo:
            subprocess.run(["gzip", "-c"], stdin=fi, stdout=fo, check=True)
        f.unlink()
        return f"✅ {f.name} → {gz.name}"
    except Exception as e:
        return f"❌ Erreur avec {f.name}: {e}"

def decompress_and_recompress_z_file(z: Path):
    try:
        gz = z.with_suffix("").with_suffix(".fits.gz")
        print(f"... {z.name} → {gz.name}")
        with subprocess.Popen(["zcat", str(z)], stdout=subprocess.PIPE) as p1, open(gz, "wb") as fo:
            subprocess.run(["gzip", "-c"], stdin=p1.stdout, stdout=fo, check=True)
        z.unlink()
        return f"✅ {z.name} → {gz.name}"
    except Exception as e:
        return f"❌ Erreur avec {z.name}: {e}"

def main(root=".", max_workers=None):
    root_path = Path(root)
    # First, rename all .fits and .fits.Z files
    for ext in ("*.fits", "*.fits.Z", "*.xml", "*.png"):
        for p in root_path.rglob(ext):
            if p.is_file() and is_visible(p):
                safe = sanitize_filename(p.name)
                if safe != p.name:
                    print(f"Renaming {p.name} to {safe}...")
                    p.rename(p.with_name(safe))

    # Now, collect files again after renaming
    z_files = [p for p in root_path.rglob("*.fits.Z") if p.is_file() and is_visible(p)]
    fits_files = [p for p in root_path.rglob("*.fits") if p.is_file() and is_visible(p) and not str(p).endswith(".fits.Z")]
    all_files = [(decompress_and_recompress_z_file, f) for f in z_files] + [(compress_fits_file, f) for f in fits_files]

    max_workers = 2#int(os.cpu_count()//1.5) #max_workers or int(os.cpu_count()//3) or 4
    max_workers = min(max_workers, len(all_files)) if max_workers else len(all_files)
    if not all_files:
        print("Aucun fichier .fits ou .fits.Z visible à traiter."); return
    print(f"🔍 {len(all_files)} fichiers à traiter en parallèle ({max_workers} threads)...\n")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(func, file) for func, file in all_files]
        for f in tqdm(futures, desc="Traitement", unit="fichier"): print(f.result())

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root)