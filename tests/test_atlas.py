import os

import nibabel as nib
import numpy as np
import pytest

import parcelview as pv


def write_annot(path, labels, names):
    ctab = np.zeros((len(names), 4), np.int32)
    ctab[:, 0] = 10 * np.arange(1, len(names) + 1)  # FreeSurfer identifies labels by colour, so each needs its own
    nib.freesurfer.write_annot(str(path), np.asarray(labels, np.int32), ctab, names, fill_ctab=True)
    return path


def write_label_gifti(path, values, table):
    lt = nib.gifti.GiftiLabelTable()
    for key, name in table.items():
        lab = nib.gifti.GiftiLabel(key=key)
        lab.label = name
        lt.labels.append(lab)
    arr = nib.gifti.GiftiDataArray(np.asarray(values, np.int32), intent='NIFTI_INTENT_LABEL', datatype='NIFTI_TYPE_INT32')
    nib.save(nib.gifti.GiftiImage(labeltable=lt, darrays=[arr]), str(path))
    return path


def test_from_arrays_validates():
    with pytest.raises(ValueError, match='unique'):
        pv.Atlas.from_arrays([0, 1], [0, 1], ['a', 'a'])
    with pytest.raises(ValueError, match='-1..1'):
        pv.Atlas.from_arrays([0, 2], [0, 1], ['a', 'b'])
    with pytest.raises(ValueError, match='integers'):
        pv.Atlas.from_arrays([0, .5], [0, 1], ['a', 'b'])
    with pytest.raises(ValueError, match='3 entries'):
        pv.Atlas.from_arrays([0, 1], [0, 1], ['a', 'b'], groups=['x', 'y', 'z'])
    a = pv.Atlas.from_arrays([0, -1], [0, 0], ['a', 'b', 'c'])
    assert a.on_surface.tolist() == [True, False, False]
    assert a.labels == a.names


def test_from_annot_order_background_and_duplicates(tmp_path):
    # left: background, A, B and an empty label C; right: background, A, D
    lh = write_annot(tmp_path / 'lh.annot', [0, 1, 1, 2, 0], ['unknown', 'A', 'B', 'C'])
    rh = write_annot(tmp_path / 'rh.annot', [2, 1, 0, -1, 1], ['Medial_Wall', 'A', 'D'])
    a = pv.Atlas.from_annot(lh, rh)
    assert a.names == ('A (left)', 'B', 'A (right)', 'D')
    assert a.left.tolist() == [-1, 0, 0, 1, -1]
    assert a.right.tolist() == [3, 2, -1, -1, 2]
    b = pv.Atlas.from_annot(lh, rh, bilateral=True)
    assert b.names == ('A', 'B', 'D')
    assert b.right.tolist() == [2, 0, -1, -1, 0]
    c = pv.Atlas.from_annot(lh, rh, background=['unknown'])  # 'Medial_Wall' now counts as a parcel
    assert 'Medial_Wall' in c.names


def test_from_gifti(tmp_path):
    lh = write_label_gifti(tmp_path / 'lh.label.gii', [0, 1, 2, 2], {0: '???', 1: 'L_V1', 2: 'L_V2'})
    rh = write_label_gifti(tmp_path / 'rh.label.gii', [3, 3, 0, 4], {0: '???', 3: 'R_V1', 4: 'R_V2'})
    a = pv.Atlas.from_gifti(lh, rh)
    assert a.names == ('L_V1', 'L_V2', 'R_V1', 'R_V2')
    assert a.left.tolist() == [-1, 0, 1, 1] and a.right.tolist() == [2, 2, -1, 3]
    mask = write_label_gifti(tmp_path / 'mask.label.gii', [0, 1, 1, 0], {0: '???'})  # 1 has no table entry
    with pytest.warns(UserWarning, match='missing from the label table'):
        m = pv.Atlas.from_gifti(mask, mask, bilateral=True)
    assert m.names == ('1',) and m.left.tolist() == [-1, 0, 0, -1]


def test_from_cifti_with_subcortex(tmp_path):
    n = 6
    left = nib.cifti2.BrainModelAxis.from_mask(np.array([1, 1, 1, 1, 0, 0], bool), name='CortexLeft')
    right = nib.cifti2.BrainModelAxis.from_mask(np.array([1, 1, 1, 1, 1, 1], bool), name='CortexRight')
    mask = np.zeros((3, 3, 3), bool)
    mask[1, 1, :2] = True
    thal = nib.cifti2.BrainModelAxis.from_mask(mask, name='ThalamusLeft', affine=np.eye(4))
    models = left + right + thal
    table = {0: ('???', (0, 0, 0, 0)), 1: ('R_A', (1, 0, 0, 1)), 2: ('L_A', (0, 1, 0, 1)),
             3: ('Thalamus', (0, 0, 1, 1)), 4: ('Unused', (1, 1, 0, 1))}
    labels = nib.cifti2.LabelAxis(['parcellation'], [table])
    data = np.array([[2, 2, 0, 2] + [1, 1, 1, 0, 1, 1] + [3, 3]], np.float32)
    path = tmp_path / 'atlas.dlabel.nii'
    nib.Cifti2Image(data, header=(labels, models)).to_filename(path)
    a = pv.Atlas.from_cifti(path)
    assert a.names == ('R_A', 'L_A', 'Thalamus')  # by key; 'Unused' reaches nothing
    assert a.left.tolist() == [1, 1, -1, 1, -1, -1]
    assert a.right.tolist() == [0, 0, 0, -1, 0, 0]
    assert a.on_surface.tolist() == [True, True, False]
    assert len(a.left) == n
    with pytest.raises(ValueError, match='fsaverage'):
        pv.view(np.zeros((2, 3)), a, tr=1.)  # 6 vertices: needs an explicit surface


def test_from_volume_projects_and_keeps_subcortex():
    affine = np.array([[2, 0, 0, -90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]], float)
    data = np.zeros((91, 109, 91), np.int16)
    x = np.arange(91) * 2 - 90
    data[(x >= -75) & (x <= -20)] = 1  # a slab through the left hemisphere
    data[(x >= 20) & (x <= 75)] = 2
    ijk = np.round(np.linalg.solve(affine, [0, -30, -40, 1])[:3]).astype(int)
    data[ijk[0] - 2:ijk[0] + 3, ijk[1] - 2:ijk[1] + 3, ijk[2] - 2:ijk[2] + 3] = 3  # brainstem, far from cortex
    img = nib.Nifti1Image(data, affine)
    a = pv.Atlas.from_volume(img, ['Background', 'left slab', 'right slab', 'brainstem'])
    assert a.names == ('left slab', 'right slab', 'brainstem')
    assert a.on_surface.tolist() == [True, True, False]
    assert (a.left == 0).mean() > .3 and (a.right == 1).mean() > .3
    assert not (a.left == 1).any() and not (a.right == 0).any()
    assert a.surface == 'fsaverage5'
    b = pv.Atlas.from_volume(img, {1: 'L', 2: 'R', 3: 'BS'})
    assert b.names == ('L', 'R', 'BS')
    with pytest.raises(ValueError, match='no name'):
        pv.Atlas.from_volume(img, {1: 'L'})


def test_reorder_rename_labels_groups(quadrants):
    a = quadrants
    with pytest.warns(UserWarning, match='not drawn'):
        b = a.reorder(['R_anterior_superior', 'L_posterior_inferior', 'Thalamus'])
    assert b.names == ('R_anterior_superior', 'L_posterior_inferior', 'Thalamus')
    assert b.on_surface.tolist() == [True, True, False]
    assert np.array_equal(b.left == 1, a.left == 0) and not (b.left == 0).any()
    assert np.array_equal(b.right == 0, a.right == 7)
    assert b.groups == ('right', 'left', '')
    with pytest.raises(ValueError, match='none of the names'):
        a.reorder(['x', 'y'])
    c = a.rename(lambda n: n.lower())
    assert c.names[0] == 'l_posterior_inferior' and c.labels[0] == 'l_posterior_inferior'
    d = a.with_labels({'L_posterior_inferior': 'LPI'})
    assert d.labels[0] == 'LPI' and d.labels[1] == 'L_posterior_superior'
    assert a.with_groups(None).groups is None
    assert a.with_groups(lambda n: n[0]).groups[:5] == ('L', 'L', 'L', 'L', 'R')


def test_schaefer_names():
    assert pv.schaefer_network('7Networks_LH_Vis_1') == 'Visual'
    assert pv.schaefer_network('7Networks_RH_SalVentAttn_Med_2') == 'Salience / ventral attention'
    assert pv.schaefer_network('17Networks_LH_VisCent_ExStr_1') == 'Visual central'
    assert pv.schaefer_network('17Networks_RH_ContB_PFCl_1') == 'Control B'
    assert pv.schaefer_network('Thalamus') == ''
    assert pv.schaefer_label('7Networks_LH_Cont_Par_1') == 'Cont Par 1 (LH)'
    assert pv.schaefer_label('Thalamus') == 'Thalamus'


@pytest.mark.skipif(not os.environ.get('PARCELVIEW_NETWORK'), reason='downloads; set PARCELVIEW_NETWORK=1')
def test_schaefer_download():
    a = pv.Atlas.schaefer(200, networks=17)
    assert a.n_parcels == 200 and a.on_surface.all()
    assert a.names[0].startswith('17Networks_LH_') and a.names[-1].startswith('17Networks_RH_')
    assert '' not in a.groups
