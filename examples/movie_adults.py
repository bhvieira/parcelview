"""Eight adults watching the same short film, one row each, in the Schaefer-100 parcellation.

    uv run python examples/movie_adults.py --out adults.html

In nilearn's development dataset, participants watched Pixar's "Partly Cloudy" during fMRI. This downloads eight
adults' preprocessed runs (MNI space, TR 2 s, about 6 minutes, about 6 MB each) and the volumetric Schaefer atlas,
averages the BOLD signal in each parcel, regresses out each run's confounds, band-passes it to 0.01-0.1 Hz and
z-scores it. The cortex draws all eight at once, a row each, so the parcels that follow the film change together
across people. The rows follow a hierarchical clustering of the adults (average linkage on 1 - similarity, where
two adults' similarity is the correlation of their signals in time, averaged over parcels), so alike adults sit
together. The plots open on the parcel where the adults agree most: the highest mean correlation between each
adult and the average of the others.

The data are from Richardson et al. (2018), shared on OpenNeuro as ds000228
(https://doi.org/10.18112/openneuro.ds000228.v1.0.0). If you use them, cite:
    Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018). Development of the social brain from
    age three to twelve years. Nature Communications, 9, 1027. https://doi.org/10.1038/s41467-018-03399-2
"""
import argparse

import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import parcelview as pv


def zscore(a):
    return (a - a.mean(0)) / a.std(0)


def similarity(x):
    """The correlation in time between each pair of participants, averaged over parcels; x is (participant, time,
    parcel)."""
    z = np.stack([zscore(a) for a in x])
    return np.einsum('itp,jtp->ij', z, z) / (x.shape[1] * x.shape[2])


def cluster_order(sim):
    """The leaf order of an average-linkage clustering on 1 - similarity, its branches flipped so that neighbouring
    leaves are as alike as the tree allows."""
    return leaves_list(linkage(squareform(1 - sim, checks=False), 'average', optimal_ordering=True))


def agreement(x):
    """Per parcel, the mean over participants of the correlation between one participant and the average of the
    others; x is (participant, time, parcel)."""
    return np.mean([(zscore(x[i]) * zscore(np.delete(x, i, 0).mean(0))).mean(0) for i in range(len(x))], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='adults.html')
    ap.add_argument('--n', type=int, default=8, help='how many adults, at most 8 (default: 8)')
    ap.add_argument('--data-dir', help="where nilearn keeps its downloads (default: nilearn's)")
    args = ap.parse_args()
    fmri = datasets.fetch_development_fmri(n_subjects=args.n, age_group='adult', data_dir=args.data_dir, verbose=0)
    volume = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2, data_dir=args.data_dir, verbose=0)
    tr = float(fmri.t_r)
    masker = NiftiLabelsMasker(volume.maps, lut=volume.lut, standardize='zscore_sample', detrend=True, t_r=tr,
                               high_pass=.01, low_pass=.1)
    series = {}
    for func, confounds, who in zip(fmri.func, fmri.confounds, fmri.phenotypic.itertuples()):
        series[f'{who.participant_id} ({who.Age:.0f} y)'] = masker.fit_transform(func, confounds=confounds)
    names = [str(n) for n in masker.region_names_.values()]  # the columns, in label order
    atlas = pv.Atlas.schaefer(100).reorder(names)
    x = np.stack(list(series.values()))
    sim, agree = similarity(x), agreement(x)
    order, rows = cluster_order(sim), list(series)
    for i, j in zip(order, [*order[1:], None]):
        print(rows[i], '' if j is None else f'  {sim[i, j]:.2f} with the next')
    series = {rows[i]: series[rows[i]] for i in order}
    for r in np.argsort(agree)[::-1][:5]:
        print(f'{agree[r]:.2f}  {names[r]}')
    page = pv.view(series, atlas, tr=tr, units='z', cortex='rows', selected=names[np.argmax(agree)], speed=4,
                   title=f'{len(series)} adults watching the same film',
                   description="nilearn's development dataset, watching Partly Cloudy, averaged in Schaefer-100 "
                               'parcels; confounds regressed out, band-passed to 0.01-0.1 Hz, z-scored.'
                               ' Data: Richardson et al. (2018), Nature Communications 9, 1027; OpenNeuro ds000228.')
    print(page.save(args.out), page)


if __name__ == '__main__':
    main()
