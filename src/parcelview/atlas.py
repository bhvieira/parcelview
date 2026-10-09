"""Parcellations: which parcel each surface vertex belongs to, and the parcels' names."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path

import nibabel as nib
import numpy as np

from ._fetch import cache_dir, fetch
from .surface import Surface

SCHAEFER_URL = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/'
                'Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/{mesh}/label/'
                '{hemi}.Schaefer2018_{n}Parcels_{k}Networks_order.annot')
# Yeo network abbreviations in Schaefer parcel names, longest first so that 'VisCent' wins over 'Vis'
NETWORKS = {'VisCent': 'Visual central', 'VisPeri': 'Visual peripheral', 'Vis': 'Visual', 'SomMot': 'Somatomotor',
            'DorsAttn': 'Dorsal attention', 'SalVentAttn': 'Salience / ventral attention', 'Limbic': 'Limbic',
            'Cont': 'Control', 'Default': 'Default', 'TempPar': 'Temporal parietal'}
BACKGROUND = frozenset({'', '???', 'unknown', 'background', 'medial_wall', 'medialwall', 'medial wall'})

Background = Collection[str] | Callable[[str], bool] | None


def is_background(name: str) -> bool:
    """The default rule for labels that are not parcels: 'Unknown', '???', medial wall and 'Background...' labels."""
    n = name.strip().lower()
    return n in BACKGROUND or n.startswith('background')


def _background_rule(background: Background) -> Callable[[str], bool]:
    if background is None:
        return is_background
    if callable(background):
        return background
    names = {str(b).strip().lower() for b in background}
    return lambda name: name.strip().lower() in names


def _per_parcel(values, names: Sequence[str], what: str) -> tuple[str, ...]:
    """A string per parcel from a sequence in parcel order, a mapping from name, or a function of the name."""
    if callable(values):
        out = [values(n) for n in names]
    elif isinstance(values, Mapping):
        out = [values.get(n, '') for n in names]
    else:
        out = list(values)
        if len(out) != len(names):
            raise ValueError(f'{what} has {len(out)} entries but the atlas has {len(names)} parcels')
    return tuple('' if v is None else str(v) for v in out)


@dataclass(frozen=True, eq=False)
class Atlas:
    """A parcellation of the cortical surface.

    left, right: each vertex's parcel index (0 to n_parcels - 1), or -1 for none, such as the medial wall.
    names: one per parcel, in the order of your data's columns. Parcels that no vertex belongs to,
        such as subcortical ones, are allowed: the viewer lists them but can't draw them.
    labels: how the viewer shows each parcel's name (defaults to the names).
    groups: each parcel's group, such as its network, for the parcel menu ('' for none).
    surface: the mesh to draw on: a Surface, a name ('fsaverage5', 'fsLR'), or None to infer it from the vertex
        counts (fsaverage meshes and fsLR 32k).
    """
    left: np.ndarray
    right: np.ndarray
    names: tuple[str, ...]
    labels: tuple[str, ...] | None = None
    groups: tuple[str, ...] | None = None
    surface: Surface | str | None = None

    def __post_init__(self):
        names = tuple(str(n) for n in self.names)
        if not names:
            raise ValueError('the atlas has no parcels')
        seen = set()
        dup = sorted({n for n in names if n in seen or seen.add(n)})
        if dup:
            raise ValueError(f'parcel names must be unique; repeated: {dup[:5]}')
        object.__setattr__(self, 'names', names)
        for side in ('left', 'right'):
            lab = np.asarray(getattr(self, side))
            if lab.ndim != 1 or not len(lab):
                raise ValueError(f'{side} must be a 1-D array of vertex labels, got shape {lab.shape}')
            if not np.issubdtype(lab.dtype, np.integer):
                if not np.all(np.isfinite(lab)) or np.any(lab != np.round(lab)):
                    raise ValueError(f'{side} labels must be integers')
            lab = lab.astype(np.int32)
            if lab.min() < -1 or lab.max() >= len(names):
                raise ValueError(f'{side} labels must lie in -1..{len(names) - 1}')
            object.__setattr__(self, side, lab)
        object.__setattr__(self, 'labels', names if self.labels is None else _per_parcel(self.labels, names, 'labels'))
        if self.groups is not None:
            object.__setattr__(self, 'groups', _per_parcel(self.groups, names, 'groups'))

    @property
    def n_parcels(self) -> int:
        return len(self.names)

    @property
    def on_surface(self) -> np.ndarray:
        """Whether each parcel covers at least one vertex."""
        hit = np.zeros(self.n_parcels, bool)
        for lab in (self.left, self.right):
            hit[lab[lab >= 0]] = True
        return hit

    def __repr__(self):
        drawn = int(self.on_surface.sum())
        surf = self.surface.name if isinstance(self.surface, Surface) else self.surface
        return (f'Atlas({self.n_parcels} parcels, {drawn} on the surface, {len(self.left)} + {len(self.right)} vertices'
                + (f', surface={surf!r}' if surf else '') + ')')

    # ---------- constructors ----------
    @classmethod
    def from_arrays(cls, left, right, names, labels=None, groups=None, surface=None) -> Atlas:
        """From each vertex's parcel index (-1 for none) and the parcels' names."""
        return cls(left, right, tuple(names), labels, groups, surface)

    @classmethod
    def from_annot(cls, left: str | PathLike, right: str | PathLike, *, surface=None, background: Background = None,
                   bilateral: bool = False) -> Atlas:
        """From FreeSurfer .annot files, one per hemisphere.

        Parcels are numbered left hemisphere first, each in the annotation's colour-table order, which is the
        usual order for Schaefer and Desikan-Killiany data. Labels that cover no vertex are skipped. A name that
        occurs in both hemispheres becomes two parcels, 'name (left)' and 'name (right)', unless bilateral=True
        makes it one.
        """
        hemis = []
        for path in (left, right):
            lab, _, names = nib.freesurfer.read_annot(str(path))
            hemis.append((lab, {i: n.decode() if isinstance(n, bytes) else str(n) for i, n in enumerate(names)}))
        return cls._from_tables(*hemis, background=background, bilateral=bilateral, surface=surface)

    @classmethod
    def from_gifti(cls, left: str | PathLike, right: str | PathLike, *, surface=None, background: Background = None,
                   bilateral: bool = False) -> Atlas:
        """From GIfTI label files (.label.gii), one per hemisphere; parcels are numbered as in from_annot."""
        hemis = []
        for path in (left, right):
            img = nib.load(path)
            hemis.append((np.round(np.asarray(img.agg_data())).astype(np.int64).ravel(), img.labeltable.get_labels_as_dict()))
        return cls._from_tables(*hemis, background=background, bilateral=bilateral, surface=surface)

    @classmethod
    def from_cifti(cls, path: str | PathLike, *, surface=None, background: Background = None, map_index: int = 0) -> Atlas:
        """From a CIFTI dense label file (.dlabel.nii), such as an HCP parcellation on fsLR 32k.

        Parcels are numbered by label key, the order Connectome Workbench uses when it parcellates data.
        Labels found only in the subcortical volume become parcels that are listed but not drawn. 32k files
        are drawn on fsLR's inflated mesh (downloaded once from TemplateFlow); for other meshes pass
        surface=Surface.from_files(left, right).
        """
        img = nib.load(path)
        lab_axis, models = img.header.get_axis(0), img.header.get_axis(1)
        if not isinstance(lab_axis, nib.cifti2.LabelAxis) or not isinstance(models, nib.cifti2.BrainModelAxis):
            raise ValueError(f'{path} is not a dense label (dlabel) CIFTI file')
        values = np.asarray(img.dataobj[map_index]).astype(np.int64)
        table = {int(k): v[0] for k, v in lab_axis.label[map_index].items()}
        cortex, volume = {}, []
        for name, sl, bm in models.iter_structures():
            if name in ('CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT'):
                full = np.full(bm.nvertices[name], -1, np.int64)  # -1 is never a key, so vertices outside are background
                full[bm.vertex] = values[sl]
                cortex[name] = full
            else:
                volume.append(values[sl])
        if len(cortex) != 2:
            raise ValueError(f'{path} needs both CIFTI_STRUCTURE_CORTEX_LEFT and _RIGHT')
        lv, rv = cortex['CIFTI_STRUCTURE_CORTEX_LEFT'], cortex['CIFTI_STRUCTURE_CORTEX_RIGHT']
        in_volume = set(np.unique(np.concatenate(volume)).tolist()) if volume else set()
        return cls._from_keys(lv, rv, table, in_volume, background=background, surface=surface)

    @classmethod
    def from_volume(cls, img, names: Sequence[str] | Mapping[int, str] | None = None, *, mesh: str = 'fsaverage5',
                    radius: float = 3., background: Background = None, surface=None) -> Atlas:
        """From a volumetric label image (NIfTI path or image), projected onto an fsaverage surface.

        Each vertex takes the most frequent label along its normal between the white and pial surfaces (nilearn's
        vol_to_surf). Parcels are numbered by label value. Labels that reach no vertex, such as subcortical ones,
        are listed but not drawn.

        names: a sequence indexed by label value (as nilearn atlases give, with 'Background' at 0), a mapping
            from label value to name, or None to name parcels by their values.
        mesh: the fsaverage mesh to sample on; the atlas is drawn on its inflated version unless surface is given.
        """
        from nilearn import datasets, image
        from nilearn.surface import vol_to_surf
        img = image.load_img(img)
        if img.ndim != 3:
            raise ValueError(f'the label image must be 3-D, got {img.ndim}-D')
        data = np.asarray(img.dataobj)
        if not np.issubdtype(data.dtype, np.integer) and np.any(data != np.round(data)):
            raise ValueError('the label image must hold integer labels')
        keys = sorted(int(v) for v in np.unique(data) if v != 0)
        if names is None:
            table = {k: str(k) for k in keys}
        elif isinstance(names, Mapping):
            table = {int(k): str(v) for k, v in names.items()}
        else:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
            if len(names) <= max(keys, default=0):
                raise ValueError(f'names has {len(names)} entries but the image has label {max(keys)}; '
                                 'give one name per label value, starting at 0')
            table = {k: names[k] for k in range(len(names))}
        missing = [k for k in keys if k not in table]
        if missing:
            raise ValueError(f'no name for labels {missing[:5]}')
        fs = datasets.load_fsaverage('fsaverage' if mesh == 'fsaverage7' else mesh)
        sampled = [np.round(vol_to_surf(img, fs['pial'].parts[h], inner_mesh=fs['white_matter'].parts[h],
                                        interpolation='nearest_most_frequent', radius=radius)).astype(np.int64)
                   for h in ('left', 'right')]
        sampled = [np.where(np.isin(s, keys), s, -1) for s in sampled]
        return cls._from_keys(*sampled, {k: table[k] for k in keys}, set(keys), background=background,
                              surface=surface or mesh)

    @classmethod
    def schaefer(cls, n_parcels: int = 100, networks: int = 7, mesh: str = 'fsaverage5',
                 data_dir: str | PathLike | None = None) -> Atlas:
        """The Schaefer 2018 parcellation on fsaverage, with each parcel's Yeo network as its group.

        Downloads the FreeSurfer annotations from the CBIG repository once. Names are as in the annotations
        and in nilearn's volumetric Schaefer atlas ('7Networks_LH_Vis_1'); labels are shorter ('Vis 1 (LH)').
        mesh: 'fsaverage5', 'fsaverage6' or 'fsaverage'.
        """
        if n_parcels not in range(100, 1001, 100) or networks not in (7, 17):
            raise ValueError('Schaefer parcellations have 100, 200, ..., 1000 parcels and 7 or 17 networks')
        if mesh not in ('fsaverage5', 'fsaverage6', 'fsaverage'):
            raise ValueError(f"mesh must be 'fsaverage5', 'fsaverage6' or 'fsaverage', got {mesh!r}")
        folder = Path(data_dir) if data_dir else cache_dir() / 'schaefer2018' / mesh
        paths = [fetch(SCHAEFER_URL.format(mesh=mesh, hemi=h, n=n_parcels, k=networks),
                        folder / f'{h}.Schaefer2018_{n_parcels}Parcels_{networks}Networks_order.annot')
                 for h in ('lh', 'rh')]
        atlas = cls.from_annot(*paths, surface=mesh)
        return replace(atlas, labels=tuple(schaefer_label(n) for n in atlas.names),
                       groups=tuple(schaefer_network(n) for n in atlas.names))

    @classmethod
    def _from_tables(cls, left, right, *, background, bilateral, surface) -> Atlas:
        """From two hemispheres of (values per vertex, {value: name}); see from_annot for the numbering."""
        skip = _background_rule(background)
        present = []
        for vals, table in (left, right):
            hit = set(int(v) for v in np.unique(vals))
            unlisted = sorted(k for k in hit if k > 0 and k not in table)  # 0 and negative values mean unlabelled
            if unlisted:
                warnings.warn(f'values {unlisted[:5]} are missing from the label table; they become parcels named '
                              'by their value', stacklevel=3)
                table = {**table, **{k: str(k) for k in unlisted}}
            present.append([(k, table[k]) for k in sorted(table) if k in hit and not skip(table[k])])
        both = {n for _, n in present[0]} & {n for _, n in present[1]}
        names, index, labels = [], {}, []
        for side, (vals, _), pairs in zip(('left', 'right'), (left, right), present):
            out = np.full(len(vals), -1, np.int32)
            for key, name in pairs:
                if name in both and not bilateral:
                    name = f'{name} ({side})'
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
                out[vals == key] = index[name]
            labels.append(out)
        return cls(labels[0], labels[1], tuple(names), surface=surface)

    @classmethod
    def _from_keys(cls, lv, rv, table, elsewhere, *, background, surface) -> Atlas:
        """From per-vertex keys of one label table shared by both hemispheres, numbered by key. Keys in
        `elsewhere` that reach no vertex become parcels that are not on the surface."""
        skip = _background_rule(background)
        on = set(np.unique(lv).tolist()) | set(np.unique(rv).tolist())
        keys = [k for k in sorted(table) if (k in on or k in elsewhere) and not skip(table[k])]
        remap = {k: i for i, k in enumerate(keys)}
        lab = []
        for vals in (lv, rv):
            out = np.full(len(vals), -1, np.int32)
            for k, i in remap.items():
                out[vals == k] = i
            lab.append(out)
        return cls(lab[0], lab[1], tuple(table[k] for k in keys), surface=surface)

    # ---------- edits ----------
    def reorder(self, names: Sequence[str]) -> Atlas:
        """The atlas with its parcels in the order of `names`, such as your data's column names.

        Atlas parcels missing from `names` are dropped (their vertices are drawn as unlabelled). Names that are
        not in the atlas are added as parcels that are listed but not drawn, for signals from elsewhere such as
        subcortical regions; a warning lists them, in case they are typos.
        """
        names = [str(n) for n in names]
        if len(set(names)) != len(names):
            raise ValueError('names must be unique')
        old = {n: i for i, n in enumerate(self.names)}
        extra = [n for n in names if n not in old]
        if len(extra) == len(names):
            raise ValueError(f'none of the names are in the atlas (atlas names look like {self.names[:3]})')
        if extra:
            warnings.warn(f'{len(extra)} names are not in the atlas and are added as parcels that are not drawn: '
                          f'{extra[:5]}', stacklevel=2)
        remap = np.full(self.n_parcels, -1, np.int32)
        for i, n in enumerate(names):
            if n in old:
                remap[old[n]] = i
        side = lambda lab: np.where(lab >= 0, remap[np.maximum(lab, 0)], -1)
        pick = lambda values, default: tuple(values[old[n]] if n in old else default(n) for n in names)
        return replace(self, left=side(self.left), right=side(self.right), names=tuple(names),
                       labels=pick(self.labels, lambda n: n),
                       groups=None if self.groups is None else pick(self.groups, lambda n: ''))

    def rename(self, names: Sequence[str] | Mapping[str, str] | Callable[[str], str]) -> Atlas:
        """The atlas with new names (a sequence in parcel order, a mapping from old name, or a function)."""
        new = _per_parcel(names if not isinstance(names, Mapping) else (lambda n: names.get(n, n)), self.names, 'names')
        # labels that were just the old names follow the new names
        labels = tuple(n if lab == o else lab for lab, o, n in zip(self.labels, self.names, new))
        return replace(self, names=new, labels=labels)

    def with_labels(self, labels: Sequence[str] | Mapping[str, str] | Callable[[str], str]) -> Atlas:
        """The atlas with new display labels (a sequence in parcel order, a mapping from name, or a function)."""
        new = labels if not isinstance(labels, Mapping) else (lambda n: labels.get(n, n))
        return replace(self, labels=_per_parcel(new, self.names, 'labels'))

    def with_groups(self, groups: Sequence[str] | Mapping[str, str] | Callable[[str], str] | None) -> Atlas:
        """The atlas with new groups (a sequence in parcel order, a mapping from name, a function, or None)."""
        return replace(self, groups=None if groups is None else _per_parcel(groups, self.names, 'groups'))


def schaefer_network(name: str) -> str:
    """The Yeo network of a Schaefer parcel name: '7Networks_LH_Vis_1' -> 'Visual', '17Networks_RH_ContB_PFCl_1'
    -> 'Control B'. Names that don't parse give ''."""
    parts = name.split('_')
    if len(parts) < 3:
        return ''
    abbr = parts[2]
    for key, full in NETWORKS.items():
        if abbr.startswith(key):
            rest = abbr[len(key):]
            return f'{full} {rest}' if rest else full
    return ''


def schaefer_label(name: str) -> str:
    """A short label for a Schaefer parcel name: '7Networks_LH_Vis_1' -> 'Vis 1 (LH)'."""
    parts = name.split('_')
    if len(parts) < 3 or parts[1] not in ('LH', 'RH'):
        return name
    return f"{' '.join(parts[2:])} ({parts[1]})"
