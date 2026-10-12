"""Render the README's images from the examples' pages, in the light and the dark theme.

    uv run --with playwright --with pillow python docs/render.py

This builds each example's page (the first build downloads nilearn's development dataset), opens it in headless
Chromium, and writes WebP images to docs/images. Playwright needs its browser once:
`uv run --with playwright playwright install chromium`. The cortex is drawn by SwiftShader, Chromium's software
WebGL, so the images don't depend on the GPU.
"""
import argparse
import io
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'images'
WIDTH = 1100  # wide enough to put the cortex and the plots side by side
OPENER = 'movie_fmri.py', 128., 156.  # the example, and the stretch of it that plays (s)
FPS = 10
STILLS = {  # image: the example, and the moment shown (s): 144 s is when the adults' maps agree most
    'adults': ('movie_adults.py', 144.),
    'ages': ('movie_ages.py', 144.),
}
HEIGHT = 900  # of the stills; the rest of the page is cut off
THEMES = 'light', 'dark'


def build(examples, pages: Path) -> dict:
    """Each example's page, built into `pages` unless it is already there."""
    paths = {e: pages / (Path(e).stem + '.html') for e in examples}
    jobs = [subprocess.Popen([sys.executable, str(ROOT / 'examples' / e), '--out', str(p)], stdout=subprocess.DEVNULL)
            for e, p in paths.items() if not p.exists()]
    if any(j.wait() for j in jobs):
        sys.exit('an example failed')
    return paths


def at(page, t: float):
    """Draw the page at `t` seconds, and wait until it is on screen."""
    page.evaluate('t => new Promise(done => { state.t = T0 + t; state.dirty = true; '
                  'requestAnimationFrame(() => requestAnimationFrame(done)); })', t)


def frame(page) -> Image.Image:
    return Image.open(io.BytesIO(page.screenshot())).convert('RGB')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pages', type=Path, help='where to build the pages (default: a temporary folder); pages '
                                               'already there are reused')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pages = args.pages or Path(tmp)
        pages.mkdir(parents=True, exist_ok=True)
        paths = build([OPENER[0], *(e for e, _ in STILLS.values())], pages)
        with sync_playwright() as p:
            browser = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
            for theme in THEMES:
                example, t0, t1 = OPENER
                page = browser.new_page(viewport={'width': WIDTH, 'height': HEIGHT}, color_scheme=theme)
                page.goto(paths[example].as_uri() + f'#t={t0}&paused')
                page.set_viewport_size({'width': WIDTH, 'height': page.evaluate('document.body.scrollHeight')})
                step = page.evaluate('state.speed') / FPS  # played at the page's own speed
                # Paused, so that only at() moves the time and draws, but with the button of a playing page
                page.evaluate("$('play').textContent = 'Pause'; $('play').setAttribute('aria-pressed', 'true')")
                frames = []
                for k in range(round((t1 - t0) / step)):
                    at(page, t0 + k * step)
                    frames.append(frame(page))
                frames[0].save(OUT / f'opener-{theme}.webp', save_all=True, append_images=frames[1:],
                               duration=1000 // FPS, loop=0, quality=75)
                page.close()
                for name, (example, t) in STILLS.items():
                    page = browser.new_page(viewport={'width': WIDTH, 'height': HEIGHT}, device_scale_factor=2,
                                            color_scheme=theme)
                    page.goto(paths[example].as_uri() + f'#t={t}&paused')
                    at(page, t)
                    frame(page).save(OUT / f'{name}-{theme}.webp', quality=85)
                    page.close()
            browser.close()
    for f in sorted(OUT.glob('*.webp')):
        print(f'{f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
