"""Strict COAD QC, physical 32um bins and two actual HE views; no fitted target statistics."""
import pathlib,json,gc
import numpy as np,pandas as pd,scipy.sparse as sp,anndata as ad,h5py
from .io import write_json as dump,sha256 as sha

def prepare_coad(rna, protein, gene_he_cache, protein_he_cache, output):
    ROOT = pathlib.Path(output)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "checks").mkdir(exist_ok=True)
    O = ROOT / 'inputs'
    O.mkdir(parents=True, exist_ok=True)
    if (O / 'READY.json').exists():
        raise FileExistsError('Refusing to overwrite an existing COAD input version')
    records = []
    datasets = []
    heinfo = []
    for (kind, rel, view) in [('gene', 'transcriptome/adata.h5ad', 'gene'), ('other', 'proteome/adata_codex.h5ad', 'protein')]:
        p = pathlib.Path(rna if kind == 'gene' else protein)
        a = ad.read_h5ad(p)
        a.var_names_make_unique()
        n = a.n_obs
        common = np.asarray(a.obs.codex_common).astype(int) == 1
        quality = np.asarray(a.obs.high_quality).astype(int) == 1 if kind == 'gene' else np.ones(n, bool)
        mask = common & quality
        hp = pathlib.Path(gene_he_cache if view == 'gene' else protein_he_cache)
        with h5py.File(hp) as h:
            hids = h['obs'][h['obs'].attrs.get('_index', '_index')][:].astype(str)
        ids = a.obs_names.astype(str).to_numpy()
        covered = pd.Index(hids).get_indexer(ids) >= 0
        keep = mask & covered
        co = np.asarray(a.obsm['spatial'], np.float64)
        bins = np.floor(co / 32).astype(np.int64)
        keys = bins[:, 0] * 1000000 + bins[:, 1]
        rec = dict(modality=kind, view=view, download_cells=n, quality_flag_cells=int(quality.sum()) if kind == 'gene' else None, common_flag_cells=int(common.sum()), quality_and_common_cells=int(mask.sum()), QC_cells_without_required_HE=int((mask & ~covered).sum()), QC_HE_cells=int(keep.sum()), own_nonempty_bins=int(len(np.unique(keys[keep]))), raw_file=str(p), raw_sha256=sha(p), HE_file=str(hp), HE_file_sha256=sha(hp))
        datasets.append((a, keep, keys, ids, co))
        heinfo.append((hp, hids))
        records.append(rec)
    common = np.intersect1d(np.unique(datasets[0][2][datasets[0][1]]), np.unique(datasets[1][2][datasets[1][1]]))
    assert len(common) > 128
    np.save(O / 'ids.npy', common.astype(str))
    np.save(O / 'bin_xy.npy', np.c_[common // 1000000, common % 1000000])
    coordinate_means = []
    cell_counts = []
    for (z, ((a, keep, keys, ids, co), (hp, hids), rec)) in enumerate(zip(datasets, heinfo, records)):
        kind = rec['modality']
        bi = np.searchsorted(common, keys)
        valid = keep & (bi < len(common))
        valid[valid] &= common[bi[valid]] == keys[valid]
        rows = bi[valid]
        cols = np.flatnonzero(valid)
        count = np.bincount(rows, minlength=len(common))
        assert (count > 0).all()
        group = sp.coo_matrix((1 / count[rows], (rows, cols)), shape=(len(common), len(ids))).tocsr().astype(np.float32)
        x = sp.csr_matrix(a.X, dtype=np.float32)
        assert (x.data >= 0).all() and np.isfinite(x.data).all()
        if kind == 'gene':
            total = np.asarray(x.sum(1)).ravel()
            assert (total[valid] > 0).all()
            x = (sp.diags(10000.0 / np.maximum(total, 1)) @ x).tocsr()
            x.data = np.log1p(x.data)
        y = (group @ x).toarray().astype(np.float32)
        np.save(O / (kind + '.npy'), y)
        np.save(O / (kind + '_names.npy'), np.asarray(a.var_names, str))
        np.save(O / (kind + '_cell_counts.npy'), count)
        np.savez_compressed(O / (kind + '_observation_support.npz'), cell_ids=ids[valid], bin_indices=rows, raw_rows=cols)
        pd.DataFrame({'cell_ID': ids[valid], 'bin_ID': common[rows], 'raw_row': cols}).to_csv(O / (kind + '_cell_to_bin.csv.gz'), index=False)
        coordinate_means.append(group @ co)
        cell_counts.append(count)
        lookup = pd.Index(ids).get_indexer(hids)
        ok = lookup >= 0
        hvalid = np.zeros(len(hids), bool)
        hvalid[ok] = valid[lookup[ok]]
        hbin = np.zeros(len(hids), int)
        hbin[hvalid] = bi[lookup[hvalid]]
        with h5py.File(hp) as h:
            he = np.zeros((len(common), h['obsm/he'].shape[1]), np.float32)
            hc = np.zeros(len(common), int)
            for start in range(0, len(hids), 4096):
                q = h['obsm/he'][start:start + 4096]
                v = hvalid[start:start + len(q)]
                ix = hbin[start:start + len(q)][v]
                np.add.at(he, ix, q[v])
                np.add.at(hc, ix, 1)
        assert np.array_equal(hc, count)
        he /= count[:, None]
        np.save(O / ('he_' + rec['view'] + '.npy'), he)
        rec.update(cells_in_final_common_bins=int(valid.sum()), final_common_bins=len(common), own_bins_excluded_by_other_modality=int(rec['own_nonempty_bins'] - len(common)), cells_in_own_QC_bins_outside_common=int(keep.sum() - valid.sum()), per_bin_cell_counts=dict(zip(['min', 'q25', 'median', 'q75', 'q95', 'max'], np.quantile(count, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist())), fraction_one_cell_bins=float((count == 1).mean()))
        del x, y, he
        gc.collect()
    np.save(O / 'coords.npy', np.mean(coordinate_means, axis=0).astype(np.float32))
    pd.DataFrame(records).to_csv(ROOT / 'checks/QC_FLOW.csv', index=False)
    pd.DataFrame(dict(bin_ID=common, bx=common // 1000000, by=common % 1000000, RNA_cells=cell_counts[0], CODEX_cells=cell_counts[1])).to_csv(ROOT / 'checks/BIN_OCCUPANCY.csv', index=False)
    meta = dict(dataset='coad', final_common_bins=len(common), QC=records, grid_unit='micrometre', grid_size=32, grid_origin=[0,0], independent_response_holdout=False, gene_aggregation='per-cell total10000/log1p, then bin mean', protein_aggregation='bin intensity mean')
    dump(O / 'READY.json', meta)
    config = dict(dataset='coad', representation='per_observation_normalized_before_source_fit', sections={view:dict(folder='inputs',he='he_'+view+'.npy') for view in ['gene','protein']})
    dump(ROOT / 'input_config.json',config)
    return ROOT / 'input_config.json'
