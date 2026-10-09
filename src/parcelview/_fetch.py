"""Downloads, cached on disk."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path


def cache_dir() -> Path:
    """Where downloads are kept: $PARCELVIEW_DATA, else $XDG_CACHE_HOME/parcelview, else ~/.cache/parcelview."""
    if os.environ.get('PARCELVIEW_DATA'):
        return Path(os.environ['PARCELVIEW_DATA'])
    return Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'parcelview'


def fetch(url: str, path: Path) -> Path:
    """Download `url` to `path` once; later calls return the file on disk."""
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(part, 'wb') as f:
            f.write(r.read())
    except OSError as e:
        part.unlink(missing_ok=True)
        raise OSError(f'could not download {url}: {e}') from e
    part.replace(path)
    return path
