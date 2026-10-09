"""One participant watching a short film, from nilearn's development dataset, in the Schaefer-100 parcellation.

    uv run python examples/movie_fmri.py --out movie.html

In the development dataset, participants watched Pixar's "Partly Cloudy" during fMRI. This downloads one
preprocessed run (MNI space, TR 2 s, about 6 minutes) and the volumetric Schaefer atlas, averages the BOLD signal
in each parcel, and regresses out the confounds that come with the run. The page holds two series, both z-scored:
the cleaned signal, and the same signal band-passed to 0.01-0.1 Hz.
"""
import argparse

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

import parcelview as pv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='movie.html')
    ap.add_argument('--data-dir', help="where nilearn keeps its downloads (default: nilearn's)")
    args = ap.parse_args()
    fmri = datasets.fetch_development_fmri(n_subjects=1, data_dir=args.data_dir, verbose=0)
    volume = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2, data_dir=args.data_dir, verbose=0)
    tr = 2.
    series = {}
    for name, band in (('Cleaned', {}), ('Band-passed 0.01-0.1 Hz', dict(high_pass=.01, low_pass=.1))):
        masker = NiftiLabelsMasker(volume.maps, lut=volume.lut, standardize='zscore_sample', detrend=True, t_r=tr, **band)
        series[name] = masker.fit_transform(fmri.func[0], confounds=fmri.confounds[0])
    names = [str(n) for n in masker.region_names_.values()]  # the columns, in label order
    atlas = pv.Atlas.schaefer(100).reorder(names)
    page = pv.view(series, atlas, tr=tr, units='z', speed=4, title='Watching a short film, one participant',
                   description="nilearn's development dataset, watching Partly Cloudy, averaged in Schaefer-100 "
                               'parcels; confounds regressed out, z-scored.')
    print(page.save(args.out), page)


if __name__ == '__main__':
    main()
