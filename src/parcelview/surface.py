"""Cortical surface meshes to draw parcels on."""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import nibabel as nib
import numpy as np

from ._fetch import cache_dir, fetch

# Vertices per hemisphere of the FreeSurfer fsaverage meshes, by nilearn's names
FSAVERAGE = {642: 'fsaverage3', 2562: 'fsaverage4', 10242: 'fsaverage5', 40962: 'fsaverage6', 163842: 'fsaverage'}
KINDS = {'inflated': 'inflated', 'pial': 'pial', 'white': 'white_matter', 'white_matter': 'white_matter'}
FSLR_VERTICES = 32492  # per hemisphere of HCP's fsLR 32k mesh, including the medial wall
FSLR_KINDS = ('inflated', 'veryinflated', 'midthickness')
FSLR_URL = 'https://templateflow.s3.amazonaws.com/tpl-fsLR/tpl-fsLR_den-32k_hemi-{hemi}_{kind}.surf.gii'


@dataclass(frozen=True, eq=False)
class Hemisphere:
    """One hemisphere's mesh: vertex coordinates (vertices, 3) in mm and triangles (faces, 3) of vertex indices."""
    coords: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        coords = np.asarray(self.coords, np.float32)
        faces = np.asarray(self.faces)
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 3:
            raise ValueError(f'coords must have shape (vertices, 3), got {coords.shape}')
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError(f'faces must have shape (triangles, 3), got {faces.shape}')
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError(f'faces must be integer vertex indices, got {faces.dtype}')
        if faces.min() < 0 or faces.max() >= len(coords):
            raise ValueError(f'faces index vertices outside 0..{len(coords) - 1}')
        if not np.isfinite(coords).all():
            raise ValueError('coords must be finite')
        object.__setattr__(self, 'coords', coords)
        object.__setattr__(self, 'faces', faces.astype(np.int64))

    @property
    def n_vertices(self) -> int:
        return len(self.coords)


@dataclass(frozen=True, eq=False)
class Surface:
    """A left and a right hemisphere mesh. Vertex order must match the atlas's labels."""
    left: Hemisphere
    right: Hemisphere
    name: str = 'custom'

    @classmethod
    def fsaverage(cls, mesh: str = 'fsaverage5', kind: str = 'inflated', data_dir: str | PathLike | None = None) -> Surface:
        """An fsaverage mesh from nilearn (fsaverage5 ships with nilearn; the others are downloaded once).

        mesh: 'fsaverage3' to 'fsaverage6', or 'fsaverage' (163,842 vertices per hemisphere).
        kind: 'inflated', 'pial' or 'white'.
        """
        from nilearn import datasets
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {sorted(KINDS)}, got {kind!r}')
        poly = datasets.load_fsaverage('fsaverage' if mesh == 'fsaverage7' else mesh, data_dir=data_dir)[KINDS[kind]]
        return cls(*(Hemisphere(poly.parts[h].coordinates, poly.parts[h].faces) for h in ('left', 'right')),
                   name=f'{mesh} {kind}')

    @classmethod
    def fslr(cls, kind: str = 'inflated', data_dir: str | PathLike | None = None) -> Surface:
        """HCP's fsLR 32k mesh (32,492 vertices per hemisphere), as CIFTI files use, downloaded once from
        TemplateFlow. kind: 'inflated', 'veryinflated' or 'midthickness'."""
        if kind not in FSLR_KINDS:
            raise ValueError(f'kind must be one of {FSLR_KINDS}, got {kind!r}')
        folder = Path(data_dir) if data_dir else cache_dir() / 'fslr'
        paths = [fetch(FSLR_URL.format(hemi=h, kind=kind), folder / Path(FSLR_URL.format(hemi=h, kind=kind)).name)
                 for h in ('L', 'R')]
        return cls.from_files(*paths, name=f'fsLR 32k {kind}')

    @classmethod
    def from_files(cls, left: str | PathLike, right: str | PathLike, name: str | None = None) -> Surface:
        """Meshes from GIfTI (.surf.gii) or FreeSurfer geometry files (lh.inflated), such as HCP's fsLR surfaces."""
        return cls(_read_mesh(left), _read_mesh(right), name=name or Path(left).name)

    @classmethod
    def from_arrays(cls, left: tuple, right: tuple, name: str = 'custom') -> Surface:
        """Meshes from (coords, faces) pairs."""
        return cls(Hemisphere(*left), Hemisphere(*right), name=name)


def _read_mesh(path) -> Hemisphere:
    path = Path(path)
    if path.name.endswith('.gii'):
        img = nib.load(path)
        return Hemisphere(img.agg_data('NIFTI_INTENT_POINTSET'), img.agg_data('NIFTI_INTENT_TRIANGLE'))
    coords, faces = nib.freesurfer.read_geometry(str(path))
    return Hemisphere(coords, faces)


def resolve(surface: Surface | str | None, n_left: int, n_right: int) -> Surface:
    """The surface to draw on: given, named ('fsaverage5', 'fsLR'), or inferred from the atlas's vertex counts."""
    if surface is None:
        if n_left != n_right or (n_left not in FSAVERAGE and n_left != FSLR_VERTICES):
            raise ValueError(f'the atlas has {n_left} and {n_right} vertices, which is neither an fsaverage mesh nor '
                             'fsLR 32k; pass surface=Surface.from_files(left, right)')
        surface = FSAVERAGE.get(n_left, 'fsLR')
    if isinstance(surface, str):
        surface = Surface.fslr() if surface.lower() in ('fslr', 'fslr32k', 'fslr 32k') else Surface.fsaverage(surface)
    if not isinstance(surface, Surface):
        raise TypeError(f'surface must be a Surface, a mesh name or None, got {type(surface).__name__}')
    if (surface.left.n_vertices, surface.right.n_vertices) != (n_left, n_right):
        raise ValueError(f'the surface has {surface.left.n_vertices} and {surface.right.n_vertices} vertices '
                         f'but the atlas labels {n_left} and {n_right}')
    return surface
