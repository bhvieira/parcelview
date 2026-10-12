import base64
import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import parcelview as pv


def payload(view):
    doc = view.to_html()
    m = re.search(r'<script type="application/json" id="pv-data">(.*?)</script>', doc, re.S)
    return json.loads(m.group(1))


def floats(s, dtype=np.float32):
    return np.frombuffer(base64.b64decode(s), dtype)


def test_round_trip(quadrants):
    rng = np.random.default_rng(0)
    fast, slow = rng.standard_normal((50, 8)), rng.standard_normal((10, 8))
    v = pv.view({'fast': fast, 'slow': slow}, quadrants, tr={'fast': .1, 'slow': .5}, start={'slow': 2.},
                units={'fast': '%'}, interpolation='linear', plot=['slow'], title='A <b>test</b>')
    p = payload(v)
    assert [s['name'] for s in p['signals']] == ['fast', 'slow']
    f, sl = p['signals']
    assert np.allclose(floats(f['data']).reshape(50, 8), fast.astype(np.float32))
    assert sl['tr'] == .5 and sl['start'] == 2. and f['unit'] == '%' and sl['unit'] == ''
    assert f['cmap'] == 'diverging' and f['limits'][0] == -f['limits'][1]
    assert p['interpolation'] == 'linear' and p['plot'] == [1] and p['loop'] is True
    assert p['parcels']['groups'][0] == 'left'
    assert len(p['hemis']) == 2 and p['hemis'][0]['n'] == 10242 and p['hemis'][0]['face_type'] == 'uint16'
    lab = np.frombuffer(base64.b64decode(p['hemis'][0]['parcel']), np.int16)
    assert np.array_equal(lab, quadrants.left)
    doc = v.to_html()
    assert '<title>A &lt;b&gt;test&lt;/b&gt;</title>' in doc
    assert doc.count('</script>') == 2


@pytest.mark.parametrize('label', ['</script><script>alert(1)</script>', '<!--<script>', 'a & b > c'])
def test_names_cannot_break_out_of_the_page(quadrants, label):
    v = pv.view(np.zeros((3, 8)), quadrants.with_labels(lambda n: label), tr=1.)
    data = re.search(r'id="pv-data">(.*?)</script>', v.to_html(), re.S).group(1)
    assert not set('<>&') & set(data)
    assert v.to_html().count('</script>') == 2
    assert payload(v)['parcels']['labels'][0] == label


def test_dataframe_columns_are_matched(quadrants):
    df = pd.DataFrame({'R_anterior_superior': [1., 2.], 'L_posterior_inferior': [3., 4.]})
    p = payload(pv.view(df, quadrants, tr=2.))
    d = floats(p['signals'][0]['data']).reshape(2, 8)
    assert d[:, 7].tolist() == [1., 2.] and d[:, 0].tolist() == [3., 4.] and np.isnan(d[:, 1:7]).all()
    with pytest.raises(ValueError, match='not atlas parcels'):
        pv.view(pd.DataFrame({'nope': [1.]}), quadrants, tr=1.)


class Frame:
    """A DataFrame like polars', whose to_numpy takes no dtype."""

    def __init__(self, columns):
        self.columns, self._values = list(columns), np.column_stack(list(columns.values()))

    def __len__(self):
        return len(self._values)

    def to_numpy(self):
        return self._values


def test_other_dataframes_are_matched(quadrants):
    p = payload(pv.view(Frame({'R_anterior_superior': [1, 2], 'L_posterior_inferior': [3., np.nan]}), quadrants, tr=2.))
    d = floats(p['signals'][0]['data']).reshape(2, 8)
    assert d[:, 7].tolist() == [1., 2.] and d[0, 0] == 3. and np.isnan(d[1, 0])
    # pandas' nullable columns become NaN only when to_numpy is given the dtype
    df = pd.DataFrame({'L_posterior_inferior': pd.array([1., None], dtype='Float64'), 'R_anterior_superior': [2., 3.]})
    d = floats(payload(pv.view(df, quadrants, tr=2.))['signals'][0]['data']).reshape(2, 8)
    assert d[0, 0] == 1. and np.isnan(d[1, 0]) and d[:, 7].tolist() == [2., 3.]


def test_square_arrays_warn(quadrants):
    with pytest.warns(UserWarning, match='orientation cannot be checked') as record:
        pv.view({'a': np.zeros((8, 8)), 'b': np.zeros((5, 8))}, quadrants, tr=1.)
    assert len(record) == 1 and "'a'" in str(record[0].message) and record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pv.view(pd.DataFrame(np.zeros((8, 8)), columns=quadrants.names), quadrants, tr=1.)


def test_defaults(quadrants):
    x = np.zeros((20, 8))
    x[:, 5] = np.sin(np.arange(20))  # the most variable parcel shows first
    p = payload(pv.view(x, quadrants, tr=1.))
    assert p['selected'] == 5 and p['signals'][0]['name'] == 'signal'
    assert p['interpolation'] == 'lanczos' and p['plot'] == [0]
    pos = payload(pv.view(np.abs(x) + 1, quadrants, tr=1.))['signals'][0]
    assert pos['cmap'] == 'sequential' and pos['limits'][0] < pos['limits'][1]
    p = payload(pv.view(x, quadrants, tr=1., limits=(-2, 3), cmaps='sequential', selected='L_posterior_superior'))
    assert p['signals'][0]['limits'] == [-2., 3.] and p['signals'][0]['cmap'] == 'sequential' and p['selected'] == 1
    nan = np.full((5, 8), np.nan)
    assert payload(pv.view(nan, quadrants, tr=1.))['signals'][0]['limits'] == [-1., 1.]


def test_rows_share_a_colour_range(quadrants):
    rng = np.random.default_rng(0)
    small, large = rng.standard_normal((40, 8)), 10 * rng.standard_normal((40, 8))
    p = payload(pv.view({'small': small, 'large': large}, quadrants, tr=1.))
    assert p['cortex'] == 'one' and p['signals'][0]['limits'] != p['signals'][1]['limits']
    p = payload(pv.view({'small': small, 'large': large}, quadrants, tr=1., cortex='rows'))
    assert p['cortex'] == 'rows' and p['signals'][0]['limits'] == p['signals'][1]['limits']
    assert p['signals'][0]['limits'][1] > np.percentile(np.abs(small), 99)
    p = payload(pv.view({'small': small, 'large': large}, quadrants, tr=1., cortex='rows', limits={'small': (-1, 1)}))
    assert p['signals'][0]['limits'] == [-1., 1.] and p['signals'][1]['limits'][1] > 1


def test_many_signals_overlay_with_hue(quadrants):
    rng = np.random.default_rng(1)
    signals = {f's{i}': (i + 1) * rng.standard_normal((20, 8)) for i in range(12)}
    p = payload(pv.view(signals, quadrants, tr=1., cortex='grid'))
    assert p['cortex'] == 'grid' and p['plots'] == 'overlay' and p['hue'] is None
    assert len({tuple(s['limits']) for s in p['signals']}) == 1  # one shared range
    ages = {f's{i}': 3. + i for i in range(12)}
    p = payload(pv.view(signals, quadrants, tr=1., cortex='grid', hue=ages, hue_label='Age (years)'))
    assert p['hue'] == {'label': 'Age (years)', 'values': [3. + i for i in range(12)]}
    assert payload(pv.view(signals, quadrants, tr=1., hue=list(ages.values())))['hue']['values'] == p['hue']['values']
    with pytest.raises(ValueError, match='missing'):
        pv.view(signals, quadrants, tr=1., hue={'s0': 1.})


@pytest.mark.parametrize('kwargs, match', [
    (dict(signals=np.zeros((5, 7))), r'shape \(time, 8\)'),
    (dict(tr=0.), 'positive'),
    (dict(tr={'other': 1.}), 'do not exist'),
    (dict(plot=['nope']), 'do not exist'),
    (dict(units={'nope': '%'}), 'do not exist'),
    (dict(signals={str(i): np.zeros((5, 8)) for i in range(9)}, plots='rows'), 'at most 8'),
    (dict(plots='stacked'), 'plots'),
    (dict(plots='overlay', plot=['signal']), 'every signal'),
    (dict(signals={'a': np.zeros((5, 8)), 'b': np.zeros((5, 8))}, plots='overlay', units={'a': '%'}), 'one unit'),
    (dict(hue=[1., 2.]), 'one number per signal'),
    (dict(hue={'nope': 1.}), 'do not exist'),
    (dict(hue=[np.nan]), 'finite'),
    (dict(limits=(1., 1.)), 'low < high'),
    (dict(interpolation='cubic'), 'interpolation'),
    (dict(cmaps='rainbow'), 'cmap'),
    (dict(cortex='mosaic'), 'cortex'),
    (dict(selected='nope'), 'not in the atlas'),
    (dict(window=0.), 'positive'),
    (dict(signals=np.full((5, 8), np.inf)), 'infinite'),
])
def test_validation(quadrants, kwargs, match):
    args = dict(signals=np.zeros((5, 8)), tr=1.) | kwargs
    with pytest.raises(ValueError, match=match):
        pv.view(args.pop('signals'), quadrants, **args)


def test_custom_surface_and_save(tmp_path, quadrants):
    coords = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    a = pv.Atlas.from_arrays([0, 0, 1, -1], [2, 2, 2, 2], ['a', 'b', 'c'],
                             surface=pv.Surface.from_arrays((coords, faces), (coords, faces)))
    v = pv.view(np.ones((4, 3)), a, tr=1.)
    path = v.save(tmp_path / 'out.html')
    assert path.read_text(encoding='utf-8') == v.to_html()
    assert v._repr_html_().startswith('<iframe srcdoc="&lt;!doctype html&gt;')
    with pytest.raises(ValueError, match='vertices'):
        pv.view(np.ones((4, 8)), quadrants, tr=1., surface=a.surface)


def test_surface_validation():
    with pytest.raises(ValueError, match='outside'):
        pv.Hemisphere(np.zeros((3, 3)), np.array([[0, 1, 3]]))
    with pytest.raises(ValueError, match='kind'):
        pv.Surface.fsaverage(kind='sphere')
