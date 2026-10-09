"""Play parcel time series on the cortex, for any parcellation, as one self-contained HTML page.

    import parcelview as pv
    atlas = pv.Atlas.schaefer(200)
    pv.view({'BOLD': bold}, atlas, tr=0.75, units='%').save('bold.html')
"""
from ._fetch import cache_dir
from .atlas import Atlas, is_background, schaefer_label, schaefer_network
from .surface import Hemisphere, Surface
from .view import View, view

__all__ = ['Atlas', 'Hemisphere', 'Surface', 'View', 'view', 'cache_dir', 'is_background', 'schaefer_label',
           'schaefer_network']
__version__ = '0.1.0'
