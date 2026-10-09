import numpy as np
import pytest

import parcelview as pv


@pytest.fixture(scope='session')
def fsaverage5():
    return pv.Surface.fsaverage('fsaverage5')


@pytest.fixture(scope='session')
def quadrants(fsaverage5):
    """Four parcels per hemisphere on fsaverage5, by quadrant of the inflated (y, z) plane; the most medial
    vertices of each hemisphere are left unlabelled."""
    labels, names = [], []
    for h, hemi in (('L', fsaverage5.left), ('R', fsaverage5.right)):
        c = hemi.coords
        q = (c[:, 1] > np.median(c[:, 1])).astype(int) * 2 + (c[:, 2] > np.median(c[:, 2]))
        offset = len(names)
        lab = q + offset
        medial = np.abs(c[:, 0]) < np.quantile(np.abs(c[:, 0]), .05)
        lab[medial] = -1
        labels.append(lab)
        names += [f'{h}_{k}' for k in ('posterior_inferior', 'posterior_superior', 'anterior_inferior', 'anterior_superior')]
    return pv.Atlas.from_arrays(labels[0], labels[1], names, groups=['left'] * 4 + ['right'] * 4)
