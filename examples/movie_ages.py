"""Children and adults watching the same short film, one brain each in a grid, in the Schaefer-100 parcellation.

    uv run python examples/movie_ages.py --out ages.html

In nilearn's development dataset, 122 children (3 to 12 years old) and 33 adults watched Pixar's "Partly Cloudy"
during fMRI. This downloads 24 children spread evenly over the children's ages and 24 adults spread over the
adults' ages (MNI space, TR 2 s, about 6 minutes, about 6 MB each, so about 290 MB) and the volumetric Schaefer
atlas. It averages the BOLD signal in each parcel, which takes a few minutes, regresses out each run's confounds,
band-passes it to 0.01-0.1 Hz and z-scores it.
The cortex draws every participant at once, youngest first, in the one view the viewer picks. The plot overlays
every participant's series in the chosen parcel, coloured by age, and opens on the parcel where the participants
agree most: the highest mean correlation between each participant and the average of the others.

nilearn's fetcher takes the first children by ID, who are the youngest, so this picks participants from its table
and downloads them with the fetcher's per-participant helper, which is private to nilearn.

The data are from Richardson et al. (2018), shared on OpenNeuro as ds000228
(https://doi.org/10.18112/openneuro.ds000228.v1.0.0). If you use them, cite:
    Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018). Development of the social brain from
    age three to twelve years. Nature Communications, 9, 1027. https://doi.org/10.1038/s41467-018-03399-2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.datasets.func import _fetch_development_fmri_functional
from nilearn.maskers import NiftiLabelsMasker

import parcelview as pv

# The confounds nilearn's fetcher keeps when it reduces them
CONFOUNDS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement',
             *[f'a_comp_cor_{i:02d}' for i in range(6)], 'csf', 'white_matter']


def spread(people, n):
    """Up to n of these people, evenly through their ages, youngest first."""
    people = people.sort_values('Age')
    return people.iloc[np.unique(np.linspace(0, len(people) - 1, n).round().astype(int))]


def fetch(n_children, n_adults, data_dir):
    """The chosen participants (rows of the dataset's table), their runs and their confounds."""
    first = datasets.fetch_development_fmri(n_subjects=1, data_dir=data_dir, verbose=0)  # brings the table
    home = Path(first.func[0]).parent
    table = pd.read_csv(home / 'participants.tsv', sep='\t')
    who = pd.concat([spread(table[table.Child_Adult == 'child'], n_children),
                     spread(table[table.Child_Adult == 'adult'], n_adults)])
    funcs, confounds = _fetch_development_fmri_functional(who, data_path=home, url=None, resume=True, verbose=0)
    return who, funcs, [pd.read_csv(c, sep='\t')[CONFOUNDS] for c in confounds]


def zscore(a):
    return (a - a.mean(0)) / a.std(0)


def agreement(x):
    """Per parcel, the mean over participants of the correlation between one participant and the average of the
    others; x is (participant, time, parcel)."""
    return np.mean([(zscore(x[i]) * zscore(np.delete(x, i, 0).mean(0))).mean(0) for i in range(len(x))], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='ages.html')
    ap.add_argument('--children', type=int, default=24, help='how many children, at most 122 (default: 24)')
    ap.add_argument('--adults', type=int, default=24, help='how many adults, at most 33 (default: 24)')
    ap.add_argument('--data-dir', help="where nilearn keeps its downloads (default: nilearn's)")
    ap.add_argument('--jobs', type=int, default=4, help='runs to average in parcels at once (default: 4)')
    args = ap.parse_args()
    who, funcs, confounds = fetch(args.children, args.adults, args.data_dir)
    volume = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2, data_dir=args.data_dir, verbose=0)
    tr = 2.
    masker = NiftiLabelsMasker(volume.maps, lut=volume.lut, standardize='zscore_sample', detrend=True, t_r=tr,
                               high_pass=.01, low_pass=.1).fit(funcs[0])  # the runs share one grid: resample once
    # nilearn takes seconds to average a run in parcels, so the runs go in parallel
    signals = Parallel(n_jobs=args.jobs)(delayed(masker.transform)(f, confounds=c) for f, c in zip(funcs, confounds))
    rows = [f"{p.removeprefix('sub-')} ({age:.1f} y)" for p, age in zip(who.participant_id, who.Age)]  # fit a grid cell
    series, ages = dict(zip(rows, signals)), dict(zip(rows, who.Age))
    names = [str(n) for n in masker.region_names_.values()]  # the columns, in label order
    atlas = pv.Atlas.schaefer(100).reorder(names)
    agree = agreement(np.stack(list(series.values())))
    for r in np.argsort(agree)[::-1][:5]:
        print(f'{agree[r]:.2f}  {names[r]}')
    kids, grown = who[who.Child_Adult == 'child'].Age, who[who.Child_Adult == 'adult'].Age
    page = pv.view(series, atlas, tr=tr, units='z', cortex='grid', hue=ages, hue_label='Age (years)',
                   selected=names[np.argmax(agree)], speed=4, title='Children and adults watching the same film',
                   description=f"nilearn's development dataset: {len(kids)} children ({int(kids.min())} to "
                               f'{int(kids.max())} years old) and {len(grown)} adults ({int(grown.min())} to '
                               f'{int(grown.max())}) watching Partly Cloudy, youngest first, averaged in Schaefer-100 '
                               'parcels; confounds regressed out, band-passed to 0.01-0.1 Hz, z-scored.'
                               ' Data: Richardson et al. (2018), Nature Communications 9, 1027; OpenNeuro ds000228.')
    print(page.save(args.out), page)


if __name__ == '__main__':
    main()
