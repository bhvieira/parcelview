"""Pack parcel time series and a surface into one self-contained, interactive HTML page."""
from __future__ import annotations

import base64
import html
import json
import tempfile
import warnings
import webbrowser
from collections.abc import Mapping, Sequence
from importlib.resources import files
from os import PathLike
from pathlib import Path

import numpy as np

from .atlas import Atlas
from .surface import Surface, resolve

MAX_ROWS = 8  # plotted in rows, each signal takes one colour in a fixed order; more would need invented colours
INTERPOLATIONS = ('lanczos', 'linear', 'nearest')
CMAPS = ('diverging', 'sequential')
CORTEX = ('one', 'rows', 'grid')
PLOTS = ('rows', 'overlay')


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _per_signal(value, names: Sequence[str], what: str, default=None) -> dict:
    """One value per signal from a single value or a mapping from signal name."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise ValueError(f'{what} names signals that do not exist: {unknown}')
        return {n: value.get(n, default) for n in names}
    return {n: default if value is None else value for n in names}


def _hue(hue, names: Sequence[str]) -> list[float]:
    """One finite number per signal, from a mapping from signal name or a sequence in signal order."""
    if isinstance(hue, Mapping):
        hue = {str(k): v for k, v in hue.items()}
        unknown, missing = sorted(set(hue) - set(names)), [n for n in names if n not in hue]
        if unknown:
            raise ValueError(f'hue names signals that do not exist: {unknown}')
        if missing:
            raise ValueError(f'hue needs a number for every signal, missing {missing[:5]}')
        values = [hue[n] for n in names]
    else:
        values = list(hue)
        if len(values) != len(names):
            raise ValueError(f'hue needs one number per signal ({len(names)}), got {len(values)}')
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('hue values must be finite numbers')
    return values.tolist()


def _as_matrix(x, atlas: Atlas, name: str) -> np.ndarray:
    """A signal as (time, parcel) in atlas order. A DataFrame's columns are matched to the atlas names."""
    if hasattr(x, 'columns') and hasattr(x, 'to_numpy'):
        cols = [str(c) for c in x.columns]
        index = {n: i for i, n in enumerate(atlas.names)}
        unknown = [c for c in cols if c not in index]
        if unknown:
            raise ValueError(f'signal {name!r} has columns that are not atlas parcels: {unknown[:5]}')
        try:
            values = x.to_numpy(dtype=float)
        except TypeError:  # polars' to_numpy takes no dtype
            values = np.asarray(x.to_numpy(), dtype=float)
        out = np.full((len(x), atlas.n_parcels), np.nan, np.float32)
        out[:, [index[c] for c in cols]] = values
        return out
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1 and atlas.n_parcels == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[1] != atlas.n_parcels:
        raise ValueError(f'signal {name!r} must have shape (time, {atlas.n_parcels}) to match the atlas, got {a.shape}')
    if len(a) < 1:
        raise ValueError(f'signal {name!r} has no samples')
    if np.isinf(a).any():
        raise ValueError(f'signal {name!r} has infinite values (use NaN for missing ones)')
    return a.astype(np.float32)


def _limits(a: np.ndarray, cmap: str | None) -> tuple[str, tuple[float, float]]:
    """A colour map and range: the 1st to 99th percentile, or for signals with both signs up to the 98th
    percentile of the absolute value, symmetric about 0."""
    finite = a[np.isfinite(a)]
    if not finite.size:
        return cmap or 'diverging', (-1., 1.)
    lo, hi = np.percentile(finite, [1, 99])
    if cmap is None:
        cmap = 'diverging' if lo < 0 < hi else 'sequential'
    if cmap == 'diverging':
        m = float(np.percentile(np.abs(finite), 98)) or float(np.abs(finite).max()) or 1.
        return cmap, (-m, m)
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        lo, hi = lo - .5, hi + .5
    return cmap, (float(lo), float(hi))


def view(signals, atlas: Atlas, *, tr: float | Mapping[str, float], start: float | Mapping[str, float] = 0.,
         units: str | Mapping[str, str] = '', interpolation: str = 'lanczos',
         limits: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
         cmaps: str | Mapping[str, str] | None = None, plot: Sequence[str] | None = None,
         plots: str | None = None, hue: Mapping[str, float] | Sequence[float] | None = None, hue_label: str = '',
         cortex: str = 'one', surface: Surface | str | None = None, groups=None, selected: str | int | None = None,
         title: str = 'Parcel time series', description: str | None = None,
         window: float = 60., speed: float = 1., loop: bool = True) -> View:
    """An interactive page that plays parcel time series on the cortex, with the time series of a chosen parcel.

    signals: a (time, parcel) array, or a mapping from name to such arrays. Columns follow the atlas's parcel
        order; a pandas or polars DataFrame is matched by column name.
    atlas: the parcellation (see Atlas).
    tr: seconds between samples, for all signals or per signal; signals may be sampled at different rates.
    start: time of each signal's first sample (s).
    units: the signals' unit, such as '%', for all or per signal.
    interpolation: how values between samples are drawn, on the cortex and in the plots: 'lanczos'
        (band-limited: smooth, and through every sample), 'linear' or 'nearest'. The viewer can change it.
    limits: the cortex's colour range, for all signals or per signal; by default the 1st to 99th percentile,
        or for signals with both signs +-the 98th percentile of the absolute value (of all signals together
        when the cortex draws them all, in rows or a grid).
    cmaps: 'diverging' (blue, gray at 0, red) or 'sequential', for all or per signal; by default diverging for
        signals with both signs.
    plot: with plots='rows', the signals plotted when the page opens; by default all of them.
    plots: how the chosen parcel's signals are plotted: 'rows', one row each, which the viewer toggles (up to 8
        signals); or 'overlay', all of them on one axis, which needs one unit. By default rows for up to 8
        signals, otherwise overlay.
    hue: a number per signal, such as each participant's age, that colours its line on a light-to-dark scale
        shown on the page; a mapping from signal name or a sequence in signal order. Without it each signal takes
        its own colour, or in an overlay all take one.
    hue_label: what hue measures, such as 'Age (years)'.
    cortex: 'one' draws one signal at a time on the cortex, chosen from a menu; 'rows' draws them all at once,
        one row each in four views, to compare them moment by moment, such as several participants who saw the
        same stimulus; 'grid' draws them all at once in one view each, for many signals. The viewer picks the
        grid's view.
    surface: the mesh to draw on, overriding the atlas's.
    groups: the parcel menu's groups, overriding the atlas's (a sequence in parcel order or a mapping from name).
    selected: the parcel plotted first (name or index); by default the most variable one.
    window: seconds of history in the plots.
    speed: initial playback speed (times real time).
    loop: whether playback starts over at the end (the viewer can change it).
    """
    if not isinstance(atlas, Atlas):
        raise TypeError(f'atlas must be an Atlas, got {type(atlas).__name__}')
    if not isinstance(signals, Mapping):
        signals = {'signal': signals}
    names = [str(n) for n in signals]
    if not names:
        raise ValueError('give at least one signal')
    if not (np.isfinite(window) and window > 0 and np.isfinite(speed) and speed > 0):
        raise ValueError('window and speed must be positive')
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f'interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}')
    if cortex not in CORTEX:
        raise ValueError(f'cortex must be one of {CORTEX}, got {cortex!r}')
    if plots is None:
        plots = 'rows' if len(names) <= MAX_ROWS else 'overlay'
    if plots not in PLOTS:
        raise ValueError(f'plots must be one of {PLOTS}, got {plots!r}')
    if plots == 'rows' and len(names) > MAX_ROWS:
        raise ValueError(f"at most {MAX_ROWS} signals can be plotted in rows, got {len(names)}; use plots='overlay'")
    if plots == 'overlay' and plot is not None:
        raise ValueError("plot chooses the rows to plot; with plots='overlay' every signal is plotted")
    hues = None if hue is None else _hue(hue, names)
    plot = names if plot is None else [str(n) for n in plot]
    unknown = sorted(set(plot) - set(names))
    if unknown:
        raise ValueError(f'plot names signals that do not exist: {unknown}')
    data = {n: _as_matrix(x, atlas, n) for n, x in zip(names, signals.values())}

    trs, starts = _per_signal(tr, names, 'tr'), _per_signal(start, names, 'start', 0.)
    for n in names:
        if trs[n] is None or not np.isfinite(trs[n]) or trs[n] <= 0:
            raise ValueError(f'tr of {n!r} must be a positive number of seconds, got {trs[n]}')
        if not np.isfinite(starts[n]):
            raise ValueError(f'start of {n!r} must be finite')
    unit = _per_signal(units, names, 'units', '')
    if plots == 'overlay' and len(set(unit.values())) > 1:
        raise ValueError(f'an overlay plots every signal on one axis, so they need one unit, '
                         f'got {sorted(set(unit.values()))}')
    cmap = _per_signal(cmaps, names, 'cmaps')
    if isinstance(limits, tuple) or (limits is not None and not isinstance(limits, Mapping)):
        limits = {n: limits for n in names}
    limit = _per_signal(limits, names, 'limits')

    # Signals drawn side by side are compared with each other, so by default they share one colour range
    pooled = np.concatenate([d.ravel() for d in data.values()]) if cortex != 'one' else None
    out = []
    for n in names:
        if cmap[n] is not None and cmap[n] not in CMAPS:
            raise ValueError(f'cmap of {n!r} must be one of {CMAPS}, got {cmap[n]!r}')
        kind, auto = _limits(data[n] if pooled is None else pooled, cmap[n])
        lo, hi = auto if limit[n] is None else map(float, limit[n])
        if not hi > lo:
            raise ValueError(f'limits of {n!r} must be (low, high) with low < high, got {limit[n]}')
        out.append(dict(name=n, n=len(data[n]), tr=float(trs[n]), start=float(starts[n]), data=_b64(data[n]),
                        unit=str(unit[n]), cmap=kind, limits=[lo, hi]))

    surf = resolve(surface if surface is not None else atlas.surface, len(atlas.left), len(atlas.right))
    group = atlas.groups if groups is None else atlas.with_groups(groups).groups
    drawn = atlas.on_surface
    if selected is None:
        first = data[names[0]]
        with warnings.catch_warnings():  # all-missing parcels have no variance
            warnings.simplefilter('ignore', RuntimeWarning)
            var = np.nan_to_num(np.nanvar(first, axis=0), nan=0.)
        selected = int(np.argmax(np.where(drawn, var, -1)))
    elif isinstance(selected, str):
        if selected not in atlas.names:
            raise ValueError(f'selected parcel {selected!r} is not in the atlas')
        selected = atlas.names.index(selected)
    elif not 0 <= int(selected) < atlas.n_parcels:
        raise ValueError(f'selected must be a parcel index in 0..{atlas.n_parcels - 1}')

    parcel_type = 'int16' if atlas.n_parcels < 32767 else 'int32'
    hemis = []
    for hemi, lab in ((surf.left, atlas.left), (surf.right, atlas.right)):
        face_type = 'uint16' if hemi.n_vertices < 65536 else 'uint32'
        hemis.append(dict(n=hemi.n_vertices, positions=_b64(hemi.coords.astype(np.float32)),
                          faces=_b64(hemi.faces.astype(face_type)), face_type=face_type,
                          parcel=_b64(lab.astype(parcel_type)), parcel_type=parcel_type))
    payload = dict(title=str(title), description=None if description is None else str(description),
                   window=float(window), speed=float(speed), loop=bool(loop), interpolation=interpolation,
                   cortex=cortex, plots=plots, hue=None if hues is None else dict(label=str(hue_label), values=hues),
                   selected=int(selected), plot=[names.index(n) for n in plot],
                   parcels=dict(names=list(atlas.names), labels=list(atlas.labels),
                                groups=None if group is None else list(group), drawn=drawn.tolist()),
                   signals=out, hemis=hemis)
    page = files('parcelview').joinpath('viewer.html').read_text(encoding='utf-8')
    # No '<', '>' or '&' inside the script element, so no name can end it early or confuse the HTML parser
    data = json.dumps(payload, separators=(',', ':'))
    data = data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    doc = page.replace('__TITLE__', html.escape(str(title))).replace('__DATA__', data)
    size = len(doc) / 1e6
    if size > 100:
        warnings.warn(f'the page is {size:.0f} MB; browsers may load it slowly (shorten or downsample the signals)',
                      stacklevel=2)
    return View(doc)


class View:
    """A rendered page. Save it with save(), open it with open(), or show it in a notebook by leaving it last
    in a cell."""

    def __init__(self, document: str):
        self._html = document

    def to_html(self) -> str:
        """The page as one HTML string."""
        return self._html

    def save(self, path: str | PathLike) -> Path:
        """Write the page to `path` (a self-contained .html file that works offline)."""
        path = Path(path)
        path.write_text(self._html, encoding='utf-8')
        return path

    def open(self) -> Path:
        """Save the page to a temporary file and open it in the default browser."""
        with tempfile.NamedTemporaryFile('w', suffix='.html', prefix='parcelview-', delete=False, encoding='utf-8') as f:
            f.write(self._html)
        path = Path(f.name)
        webbrowser.open(path.as_uri())
        return path

    def _repr_html_(self) -> str:
        return (f'<iframe srcdoc="{html.escape(self._html, quote=True)}" style="width:100%;height:1080px;border:0" '
                'title="parcelview"></iframe>')

    def __repr__(self):
        return f'<parcelview.View, {len(self._html) / 1e6:.1f} MB>'
