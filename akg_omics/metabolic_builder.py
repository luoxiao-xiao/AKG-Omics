import os,re,pickle,hashlib,json,math,xml.etree.ElementTree as ET
from collections import defaultdict,Counter
from bisect import bisect_left,bisect_right
import numpy as np,pandas as pd
KB_ROOT=HMDB_DIR=REACTOME_DIR=KB_CACHE_DIR=None
ION_MODE='positive';PPM_TOL=5.;PPM_SIGMA=3.;TOPK_CANDIDATES=2;MIN_CANDIDATE_WEIGHT=.02
USE_CHEBI_RELATION=True;MIN_SHARED_PATHWAY=1
KB_DIRECT_WEIGHT=1.;KB_PATHWAY_WEIGHT=.4
KB_GENE_GRAPH_PATHWAY_WEIGHT=1.;KB_GENE_GRAPH_HMDB_WEIGHT=.3
KB_MET_GRAPH_PATHWAY_WEIGHT=1.;KB_MET_GRAPH_CHEBI_WEIGHT=.25;KB_MET_GRAPH_HMDB_WEIGHT=.35
ADDUCTS_POS = [('M+H', 1.007276, 1, 1.0), ('M+Na', 22.989218, 1, 0.55), ('M+K', 38.963158, 1, 0.25), ('M+NH4', 18.033823, 1, 0.2)]
ADDUCTS_NEG = [('M-H', -1.007276, 1, 1.0), ('M+Cl', 34.969402, 1, 0.2), ('M+FA-H', 44.998201, 1, 0.1)]

def stable_hash(*items):
    raw = '||'.join([str(x) for x in items])
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]

def atomic_pickle_dump(obj, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def local_name(tag):
    return tag.split('}')[-1] if isinstance(tag, str) else str(tag)

def normalize_text(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == '' or x.lower() == 'nan':
        return None
    return x

def normalize_token(x):
    x = normalize_text(x)
    if x is None:
        return None
    x = x.upper().strip()
    x = re.sub('\\s+', ' ', x)
    return x

def normalize_alias(x):
    x = normalize_token(x)
    if x is None:
        return None
    x = re.sub('[^A-Z0-9]+', '', x)
    return x or None

def canonical_chebi_id(x):
    x = normalize_text(x)
    if x is None:
        return None
    m = re.search('(CHEBI:)?(\\d+)', x.upper())
    if not m:
        return None
    return f'CHEBI:{m.group(2)}'

def canonical_hmdb_id(x):
    x = normalize_text(x)
    if x is None:
        return None
    m = re.search('HMDB\\d+', x.upper())
    return m.group(0) if m else None

def canonical_ensembl_gene(x):
    x = normalize_text(x)
    if x is None:
        return None
    x = x.split('.')[0].upper()
    return x if x.startswith('ENS') else None

def canonical_ncbi_gene(x):
    x = normalize_text(x)
    if x is None:
        return None
    x = x.strip()
    return x if x.isdigit() else None

def row_normalize_matrix(m, eps=1e-08):
    m = np.asarray(m, dtype=np.float32)
    row_sum = m.sum(axis=1, keepdims=True)
    row_sum[row_sum < eps] = 1.0
    return m / row_sum

def normalize_adjacency_with_selfloop(a, eps=1e-08):
    a = np.asarray(a, dtype=np.float32)
    if a.shape[0] == 0:
        return a
    a = a + np.eye(a.shape[0], dtype=np.float32)
    deg = a.sum(axis=1, keepdims=True)
    deg[deg < eps] = 1.0
    return a / deg

def _extract_float(vals, keys):
    for k in keys:
        for x in vals.get(k, []):
            try:
                v = float(str(x))
                if np.isfinite(v):
                    return v
            except Exception:
                continue
    return None

def load_reactome_knowledge(reactome_dir):
    cache_path = os.path.join(KB_CACHE_DIR, 'reactome_maps.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    chebi_to_pathways = defaultdict(set)
    ens_to_pathways = defaultdict(set)
    ncbi_to_pathways = defaultdict(set)
    chebi_path = os.path.join(reactome_dir, 'ChEBI2Reactome_All_Levels.txt')
    ens_path = os.path.join(reactome_dir, 'Ensembl2Reactome_All_Levels.txt')
    ncbi_path = os.path.join(reactome_dir, 'NCBI2Reactome_All_Levels.txt')
    if os.path.exists(chebi_path):
        df = pd.read_csv(chebi_path, sep='\t', header=None, low_memory=False)
        for (_, row) in df.iterrows():
            cid = canonical_chebi_id(row.iloc[0])
            pw = normalize_text(row.iloc[1])
            species = normalize_text(row.iloc[5]) if len(row) > 5 else None
            if cid is None or pw is None:
                continue
            if species is not None and 'HOMO SAPIENS' not in species.upper() and ('MUS MUSCULUS' not in species.upper()):
                continue
            chebi_to_pathways[cid].add(pw)
    if os.path.exists(ens_path):
        df = pd.read_csv(ens_path, sep='\t', header=None, low_memory=False)
        for (_, row) in df.iterrows():
            gid = canonical_ensembl_gene(row.iloc[0])
            pw = normalize_text(row.iloc[1])
            species = normalize_text(row.iloc[5]) if len(row) > 5 else None
            if gid is None or pw is None:
                continue
            if species is not None and 'HOMO SAPIENS' not in species.upper() and ('MUS MUSCULUS' not in species.upper()):
                continue
            ens_to_pathways[gid].add(pw)
    if os.path.exists(ncbi_path):
        df = pd.read_csv(ncbi_path, sep='\t', header=None, low_memory=False)
        for (_, row) in df.iterrows():
            gid = canonical_ncbi_gene(row.iloc[0])
            pw = normalize_text(row.iloc[1])
            species = normalize_text(row.iloc[5]) if len(row) > 5 else None
            if gid is None or pw is None:
                continue
            if species is not None and 'HOMO SAPIENS' not in species.upper() and ('MUS MUSCULUS' not in species.upper()):
                continue
            ncbi_to_pathways[gid].add(pw)
    out = (chebi_to_pathways, ens_to_pathways, ncbi_to_pathways)
    atomic_pickle_dump(out, cache_path)
    return out

def load_hmdb_metabolite_records(hmdb_metabolites_xml):
    cache_path = os.path.join(KB_CACHE_DIR, 'hmdb_metabolites_records.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    alias_to_records = defaultdict(set)
    records = []
    mass_values = []
    if not os.path.exists(hmdb_metabolites_xml):
        return (records, alias_to_records, np.array([], dtype=np.float64), np.array([], dtype=np.int32))
    for (event, elem) in ET.iterparse(hmdb_metabolites_xml, events=('end',)):
        if local_name(elem.tag) != 'metabolite':
            continue
        vals = defaultdict(list)
        for child in elem.iter():
            nm = local_name(child.tag)
            txt = normalize_text(child.text)
            if txt is not None:
                vals[nm].append(txt)
        hmdb_id = None
        for k in ['accession', 'hmdb_id']:
            for x in vals.get(k, []):
                hm = canonical_hmdb_id(x)
                if hm is not None:
                    hmdb_id = hm
                    break
            if hmdb_id is not None:
                break
        aliases = set()
        chebi_ids = set()
        gene_symbols = set()
        pathway_ids = set()
        exact_mass = _extract_float(vals, ['monisotopic_molecular_weight', 'monisotopic_weight', 'exact_mass', 'average_molecular_weight'])
        for x in vals.get('name', []):
            aliases.add(normalize_alias(x))
            aliases.add(normalize_token(x))
        for x in vals.get('synonym', []):
            aliases.add(normalize_alias(x))
            aliases.add(normalize_token(x))
        if hmdb_id is not None:
            aliases.add(hmdb_id)
            aliases.add(normalize_alias(hmdb_id))
        for x in vals.get('chebi_id', []):
            cid = canonical_chebi_id(x)
            if cid is not None:
                chebi_ids.add(cid)
                aliases.add(cid)
                aliases.add(normalize_alias(cid))
        for x in vals.get('kegg_id', []):
            tok = normalize_token(x)
            if tok is not None:
                aliases.add(tok)
                aliases.add(normalize_alias(tok))
                aliases.add(f'KEGG COMPOUND:{tok}')
        for x in vals.get('pathway_name', []):
            pathway_ids.add(normalize_text(x))
        for k in ['gene_name', 'gene_symbol', 'protein_accession', 'protein_name', 'name']:
            for x in vals.get(k, []):
                tok = normalize_token(x)
                if tok is not None and len(tok) <= 40:
                    gene_symbols.add(tok)
        aliases = {x for x in aliases if x is not None}
        gene_symbols = {x for x in gene_symbols if x is not None}
        pathway_ids = {x for x in pathway_ids if x is not None}
        rid = len(records)
        records.append({'hmdb_id': hmdb_id, 'aliases': aliases, 'chebi_ids': chebi_ids, 'genes': gene_symbols, 'pathways': pathway_ids, 'exact_mass': exact_mass})
        if exact_mass is not None:
            mass_values.append((float(exact_mass), rid))
        for a in aliases:
            alias_to_records[a].add(rid)
        elem.clear()
    if mass_values:
        mass_values = sorted(mass_values, key=lambda x: x[0])
        mass_arr = np.array([x[0] for x in mass_values], dtype=np.float64)
        rid_arr = np.array([x[1] for x in mass_values], dtype=np.int32)
    else:
        mass_arr = np.array([], dtype=np.float64)
        rid_arr = np.array([], dtype=np.int32)
    out = (records, alias_to_records, mass_arr, rid_arr)
    atomic_pickle_dump(out, cache_path)
    return out

def get_adducts():
    if ION_MODE == 'positive':
        return ADDUCTS_POS
    if ION_MODE == 'negative':
        return ADDUCTS_NEG
    return ADDUCTS_POS + ADDUCTS_NEG

def ppm_error(obs, theo):
    return abs(obs - theo) / max(theo, 1e-12) * 1000000.0

def candidate_weight(ppm, adduct_prior):
    return float(np.exp(-ppm ** 2 / (2.0 * PPM_SIGMA ** 2))) * float(adduct_prior)

def annotate_feature_mz_to_hmdb(mz_value, mass_arr, rid_arr, ppm_tol=None, topk_candidates=None, min_candidate_weight=None):
    candidates = []
    mz = float(mz_value)
    ppm_tol = float(PPM_TOL if ppm_tol is None else ppm_tol)
    topk_candidates = int(TOPK_CANDIDATES if topk_candidates is None else topk_candidates)
    min_candidate_weight = float(MIN_CANDIDATE_WEIGHT if min_candidate_weight is None else min_candidate_weight)
    if mass_arr.size == 0:
        return candidates
    for (adduct_name, delta, charge_abs, adduct_prior) in get_adducts():
        neutral_mass = mz * charge_abs - delta
        tol = neutral_mass * ppm_tol * 1e-06
        lo = bisect_left(mass_arr, neutral_mass - tol)
        hi = bisect_right(mass_arr, neutral_mass + tol)
        for idx in range(lo, hi):
            rid = int(rid_arr[idx])
            rec_mass = float(mass_arr[idx])
            ppm = ppm_error(neutral_mass, rec_mass)
            w = candidate_weight(ppm, adduct_prior)
            if w < min_candidate_weight:
                continue
            candidates.append((w, ppm, adduct_name, rid))
    if not candidates:
        return []
    merged = {}
    for (w, ppm, adduct_name, rid) in candidates:
        if rid not in merged or w > merged[rid][0]:
            merged[rid] = (w, ppm, adduct_name)
    out = [(rid, vals[0], vals[1], vals[2]) for (rid, vals) in merged.items()]
    out.sort(key=lambda x: (-x[1], x[2]))
    return out[:topk_candidates]

def build_gene_path_sets_from_ids(gene_id_list, ens_to_pathways, ncbi_to_pathways):
    gene_infos = []
    for gid in gene_id_list:
        raw = normalize_text(gid)
        ens = canonical_ensembl_gene(raw)
        ncbi = canonical_ncbi_gene(raw)
        pathways = set()
        if ens is not None:
            pathways.update(ens_to_pathways.get(ens, set()))
        if ncbi is not None:
            pathways.update(ncbi_to_pathways.get(ncbi, set()))
        gene_infos.append({'raw': raw, 'ensembl': ens, 'ncbi': ncbi, 'pathways': pathways})
    return gene_infos

def build_full_metabolite_kb(metabolite_mz_names, gene_id_list, gene_name_list=None, enabled_sources=None, cache_tag='default', kb_precision_params=None):
    enabled = set([str(x).strip().lower() for x in enabled_sources or ['hmdb', 'reactome', 'chebi'] if str(x).strip()])
    precision_params = dict(kb_precision_params or {})
    hmdb_ppm_tol = float(precision_params.get('hmdb_ppm_tol', PPM_TOL))
    hmdb_topk_candidates = int(precision_params.get('hmdb_topk_candidates', TOPK_CANDIDATES))
    hmdb_min_candidate_weight = float(precision_params.get('hmdb_min_candidate_weight', MIN_CANDIDATE_WEIGHT))
    use_hmdb = 'hmdb' in enabled
    use_reactome = 'reactome' in enabled
    use_chebi = 'chebi' in enabled and bool(USE_CHEBI_RELATION) and use_reactome
    cache_key = stable_hash('full_met_kb_v3_agent', cache_tag, sorted(list(map(str, metabolite_mz_names))), sorted(list(map(str, gene_id_list))), sorted(list(map(str, gene_name_list or []))), sorted(list(enabled)), ION_MODE, hmdb_ppm_tol, hmdb_topk_candidates, hmdb_min_candidate_weight, USE_CHEBI_RELATION, KB_DIRECT_WEIGHT, KB_PATHWAY_WEIGHT, KB_GENE_GRAPH_PATHWAY_WEIGHT, KB_GENE_GRAPH_HMDB_WEIGHT, KB_MET_GRAPH_PATHWAY_WEIGHT, KB_MET_GRAPH_CHEBI_WEIGHT, KB_MET_GRAPH_HMDB_WEIGHT)
    cache_path = os.path.join(KB_CACHE_DIR, f'full_met_kb_{cache_key}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    if use_reactome:
        (chebi_to_pathways, ens_to_pathways, ncbi_to_pathways) = load_reactome_knowledge(REACTOME_DIR)
    else:
        (chebi_to_pathways, ens_to_pathways, ncbi_to_pathways) = ({}, {}, {})
    hmdb_xml_path = os.path.join(HMDB_DIR, 'hmdb_metabolites.xml')
    if use_hmdb:
        (hmdb_records, _, mass_arr, rid_arr) = load_hmdb_metabolite_records(hmdb_xml_path)
    else:
        (hmdb_records, mass_arr, rid_arr) = ([], np.array([], dtype=np.float64), np.array([], dtype=np.int32))
    gene_infos = build_gene_path_sets_from_ids(gene_id_list, ens_to_pathways, ncbi_to_pathways)
    n_gene = len(gene_infos)
    met_infos = []
    for mz_name in metabolite_mz_names:
        try:
            mz = float(str(mz_name))
        except Exception:
            mz = None
        candidates = annotate_feature_mz_to_hmdb(mz, mass_arr, rid_arr, ppm_tol=hmdb_ppm_tol, topk_candidates=hmdb_topk_candidates, min_candidate_weight=hmdb_min_candidate_weight) if mz is not None and use_hmdb else []
        total_w = sum((x[1] for x in candidates)) + 1e-12
        hmdb_weights = {x[0]: x[1] / total_w for x in candidates}
        hmdb_genes_weighted = defaultdict(float)
        met_pathways_weighted = defaultdict(float)
        chebi_ids = set()
        for (rid, w_norm) in hmdb_weights.items():
            rec = hmdb_records[rid]
            for cid in rec['chebi_ids']:
                chebi_ids.add(cid)
            for g in rec['genes']:
                hmdb_genes_weighted[normalize_token(g)] += w_norm
            for pw in rec['pathways']:
                met_pathways_weighted[pw] += w_norm
            if use_chebi:
                for cid in rec['chebi_ids']:
                    for pw in chebi_to_pathways.get(cid, set()):
                        met_pathways_weighted[pw] += 0.5 * w_norm
        met_infos.append({'mz_name': str(mz_name), 'candidate_weights': hmdb_weights, 'chebi_ids': chebi_ids, 'hmdb_gene_weights': dict(hmdb_genes_weighted), 'pathway_weights': dict(met_pathways_weighted)})
    gene_token_to_idx = {}
    for (idx, gid) in enumerate(gene_id_list):
        tok = normalize_token(gid)
        if tok is not None:
            gene_token_to_idx[tok] = idx
    if gene_name_list is not None:
        for (idx, gname) in enumerate(gene_name_list):
            tok = normalize_token(gname)
            if tok is not None and tok not in gene_token_to_idx:
                gene_token_to_idx[tok] = idx
    n_met = len(met_infos)
    M_direct = np.zeros((n_met, n_gene), dtype=np.float32)
    M_path = np.zeros((n_met, n_gene), dtype=np.float32)
    for (i, met) in enumerate(met_infos):
        for (g_tok, w) in met['hmdb_gene_weights'].items():
            j = gene_token_to_idx.get(g_tok, None)
            if j is not None:
                M_direct[i, j] += float(w)
    gene_path_sets = [g['pathways'] for g in gene_infos]
    for (i, met) in enumerate(met_infos):
        mpw = met['pathway_weights']
        if not mpw:
            continue
        for (j, gpw) in enumerate(gene_path_sets):
            if not gpw:
                continue
            shared = gpw.intersection(mpw.keys())
            if len(shared) >= MIN_SHARED_PATHWAY:
                M_path[i, j] = float(sum((mpw[pw] for pw in shared)))
    M = KB_DIRECT_WEIGHT * row_normalize_matrix(M_direct) + KB_PATHWAY_WEIGHT * row_normalize_matrix(M_path)
    if M.sum() > 0:
        M = row_normalize_matrix(M)
    M_rev = row_normalize_matrix(M.T) if M.sum() > 0 else np.zeros((n_gene, n_met), dtype=np.float32)
    G_gene_path = np.zeros((n_gene, n_gene), dtype=np.float32)
    for i in range(n_gene):
        for j in range(i + 1, n_gene):
            w = len(gene_path_sets[i].intersection(gene_path_sets[j]))
            if w > 0:
                G_gene_path[i, j] = G_gene_path[j, i] = float(w)
    G_gene_hmdb = np.zeros((n_gene, n_gene), dtype=np.float32)
    for met in met_infos:
        idx_weights = []
        for (g_tok, w) in met['hmdb_gene_weights'].items():
            j = gene_token_to_idx.get(g_tok, None)
            if j is not None:
                idx_weights.append((j, float(w)))
        for a in range(len(idx_weights)):
            for b in range(a + 1, len(idx_weights)):
                (ia, wa) = idx_weights[a]
                (ib, wb) = idx_weights[b]
                v = wa * wb
                G_gene_hmdb[ia, ib] += v
                G_gene_hmdb[ib, ia] += v
    gene_path_term = KB_GENE_GRAPH_PATHWAY_WEIGHT * G_gene_path if use_reactome else 0.0
    gene_hmdb_term = KB_GENE_GRAPH_HMDB_WEIGHT * G_gene_hmdb if use_hmdb else 0.0
    G_gene = normalize_adjacency_with_selfloop(gene_path_term + gene_hmdb_term)
    G_met_path = np.zeros((n_met, n_met), dtype=np.float32)
    G_met_hmdb = np.zeros((n_met, n_met), dtype=np.float32)
    for i in range(n_met):
        keys_i = set(met_infos[i]['pathway_weights'].keys())
        wi = met_infos[i]['candidate_weights']
        for j in range(i + 1, n_met):
            keys_j = set(met_infos[j]['pathway_weights'].keys())
            shared_path = keys_i.intersection(keys_j)
            if shared_path:
                G_met_path[i, j] = G_met_path[j, i] = float(sum((min(met_infos[i]['pathway_weights'][pw], met_infos[j]['pathway_weights'][pw]) for pw in shared_path)))
            wj = met_infos[j]['candidate_weights']
            shared_hmdb = set(wi.keys()).intersection(wj.keys())
            if shared_hmdb:
                G_met_hmdb[i, j] = G_met_hmdb[j, i] = float(sum((min(wi[r], wj[r]) for r in shared_hmdb)))
    met_path_term = KB_MET_GRAPH_PATHWAY_WEIGHT * G_met_path if use_reactome else 0.0
    met_chebi_term = KB_MET_GRAPH_CHEBI_WEIGHT * G_met_path if use_chebi else 0.0
    met_hmdb_term = KB_MET_GRAPH_HMDB_WEIGHT * G_met_hmdb if use_hmdb else 0.0
    G_met = normalize_adjacency_with_selfloop(met_path_term + met_chebi_term + met_hmdb_term)
    stats = {'num_metabolites': n_met, 'num_genes': n_gene, 'annotated_metabolites_by_mass': int(sum((len(x['candidate_weights']) > 0 for x in met_infos))), 'annotated_metabolites_with_pathways': int(sum((len(x['pathway_weights']) > 0 for x in met_infos))), 'hmdb_direct_links': int((M_direct > 0).sum()), 'pathway_links': int((M_path > 0).sum()), 'enabled_sources': sorted(list(enabled)), 'hmdb_ppm_tol': float(hmdb_ppm_tol), 'hmdb_topk_candidates': int(hmdb_topk_candidates), 'hmdb_min_candidate_weight': float(hmdb_min_candidate_weight), 'precision_profile': str(precision_params.get('precision_profile', ''))}
    out = {'metabolite_gene_prior': M.astype(np.float32), 'gene_metabolite_prior': M_rev.astype(np.float32), 'gene_gene_graph': G_gene.astype(np.float32), 'metabolite_metabolite_graph': G_met.astype(np.float32), 'stats': stats}
    atomic_pickle_dump(out, cache_path)
    return out


def configure(assets, cache):
    global KB_ROOT,HMDB_DIR,REACTOME_DIR,KB_CACHE_DIR
    KB_ROOT=str(assets);HMDB_DIR=os.path.join(KB_ROOT,"hmdb");REACTOME_DIR=os.path.join(KB_ROOT,"reactome");KB_CACHE_DIR=str(cache)
    os.makedirs(KB_CACHE_DIR,exist_ok=True)
