from __future__ import annotations
import os,re,pickle,hashlib,json,math,inspect
from collections import defaultdict,Counter
import numpy as np,pandas as pd,scipy.sparse as sp
PROTEO_KB_DIRECT_WEIGHT = 0.2
PROTEO_KB_MODULE_WEIGHT = 0.35
PROTEO_KB_CELLTYPE_WEIGHT = 1.5
PROTEO_KB_PROT_GRAPH_WEIGHT = 1.0
PROTEO_KB_GENE_GRAPH_WEIGHT = 1.0
PROTEO_KB_GG_CELLTYPE_WEIGHT = 0.5
PROTEO_KB_PP_CELLTYPE_WEIGHT = 0.5
PROTEO_KB_DOROTHEA_GRAPH_WEIGHT = 0.5
PROTEO_KB_OMNIPATH_GRAPH_WEIGHT = 0.5
PROTEO_KB_CORUM_GRAPH_WEIGHT = 0.5
PROTEO_KB_PROTEINATLAS_CELLTYPE_WEIGHT = 0.5
PROTEO_KB_KEGG_PATHWAY_WEIGHT = 0.5
PROTEO_KB_STRING_GRAPH_WEIGHT = 0.5
PROTEO_KB_STRING_MIN_SCORE = 700
PROTEO_KB_STRING_MAX_EDGES = 2000000

def normalize_text(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == '' or x.lower() == 'nan':
        return None
    return x

def normalize_adjacency_with_selfloop(a, eps=1e-08):
    a = np.asarray(a, dtype=np.float32)
    if a.shape[0] == 0:
        return a
    a = a + np.eye(a.shape[0], dtype=np.float32)
    deg = a.sum(axis=1, keepdims=True)
    deg[deg < eps] = 1.0
    return a / deg

def normalize_symbol(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == '' or x.lower() == 'nan':
        return None
    return x.upper()

def split_symbol_field(x):
    if x is None:
        return []
    x = str(x).strip()
    if x == '' or x.lower() == 'nan':
        return []
    parts = re.split('[|,;/\\s]+', x)
    out = []
    for p in parts:
        pp = normalize_symbol(p)
        if pp is not None:
            out.append(pp)
    return out

def normalize_matrix_rows(m, eps=1e-08):
    m = np.asarray(m, dtype=np.float32)
    row_sum = m.sum(axis=1, keepdims=True)
    row_sum[row_sum < eps] = 1.0
    return m / row_sum

def load_hgnc_maps(hgnc_file):
    df = pd.read_csv(hgnc_file, sep='\t', low_memory=False)
    symbol_col = 'approved_symbol' if 'approved_symbol' in df.columns else 'symbol'
    alias_to_symbol = {}
    ensembl_to_symbol = {}
    alias_cols = [c for c in ['alias_symbol', 'prev_symbol'] if c in df.columns]
    ensembl_col = 'ensembl_gene_id' if 'ensembl_gene_id' in df.columns else None
    for (_, row) in df.iterrows():
        canon = normalize_symbol(row[symbol_col])
        if canon is None:
            continue
        alias_to_symbol[canon] = canon
        for c in alias_cols:
            for a in split_symbol_field(row[c]):
                alias_to_symbol[a] = canon
        if ensembl_col is not None:
            ens = normalize_symbol(row[ensembl_col])
            if ens is not None:
                ensembl_to_symbol[ens] = canon
    manual_override = {'SMA': 'ACTA2', 'PAN-CYTOKERATIN': 'KRT18', 'PANC': 'KRT18', 'CK': 'KRT18', 'CD11C': 'ITGAX', 'CD56': 'NCAM1', 'HLA-DR': 'HLA-DRA', 'CD20': 'MS4A1', 'CD8': 'CD8A', 'CD3E': 'CD3E'}
    for (k, v) in manual_override.items():
        alias_to_symbol[k] = v
    return (alias_to_symbol, ensembl_to_symbol)

def load_uniprot_maps(uniprot_file, alias_to_symbol):
    df = pd.read_csv(uniprot_file, sep='\t', low_memory=False)
    acc_col = [c for c in ['Entry', 'Accession', 'From'] if c in df.columns][0]
    gene_primary_col = [c for c in ['Gene Names (primary)', 'Gene Names  (primary )'] if c in df.columns]
    gene_primary_col = gene_primary_col[0] if gene_primary_col else None
    gene_names_col = [c for c in ['Gene Names', 'Gene names'] if c in df.columns]
    gene_names_col = gene_names_col[0] if gene_names_col else None
    prot_name_col = next((c for c in ['Protein names', 'Protein names '] if c in df.columns), None)
    acc_to_symbols = defaultdict(set)
    alias_to_symbol_aug = dict(alias_to_symbol)
    protein_name_to_accs = defaultdict(set)
    for (_, row) in df.iterrows():
        acc = normalize_symbol(row[acc_col])
        if acc is None:
            continue
        all_names = []
        if gene_primary_col:
            all_names.extend(split_symbol_field(row[gene_primary_col]))
        if gene_names_col:
            all_names.extend(split_symbol_field(row[gene_names_col]))
        canon_list = []
        for name in all_names:
            canon = alias_to_symbol_aug.get(name, name)
            canon_list.append(canon)
            alias_to_symbol_aug[name] = canon
        for canon in canon_list:
            acc_to_symbols[acc].add(canon)
        if prot_name_col is not None:
            pname = str(row[prot_name_col]).strip()
            if pname and pname.lower() != 'nan':
                protein_name_to_accs[normalize_symbol(pname)].add(acc)
    return (acc_to_symbols, alias_to_symbol_aug, protein_name_to_accs)

def load_reactome_gene_sets(reactome_uniprot_file, reactome_ensembl_file, acc_to_symbols, ensembl_to_symbol):
    pathway_to_genes = defaultdict(set)
    if reactome_uniprot_file and os.path.exists(reactome_uniprot_file):
        df_u = pd.read_csv(reactome_uniprot_file, sep='\t', header=None, low_memory=False, encoding='latin1', quoting=3, on_bad_lines='skip')
        for (_, row) in df_u.iterrows():
            acc = normalize_symbol(row.iloc[0])
            pathway_name = str(row.iloc[3]).strip() if len(row) > 3 else None
            species = str(row.iloc[5]).strip() if len(row) > 5 else 'Homo sapiens'
            if pathway_name is None or 'Homo sapiens' not in species:
                continue
            for g in acc_to_symbols.get(acc, []):
                pathway_to_genes[pathway_name].add(g)
    if reactome_ensembl_file and os.path.exists(reactome_ensembl_file):
        df_e = pd.read_csv(reactome_ensembl_file, sep='\t', header=None, low_memory=False, encoding='latin1', quoting=3, on_bad_lines='skip')
        for (_, row) in df_e.iterrows():
            ens = normalize_symbol(row.iloc[0])
            pathway_name = str(row.iloc[3]).strip() if len(row) > 3 else None
            species = str(row.iloc[5]).strip() if len(row) > 5 else 'Homo sapiens'
            if pathway_name is None or 'Homo sapiens' not in species:
                continue
            g = ensembl_to_symbol.get(ens, None)
            if g is not None:
                pathway_to_genes[pathway_name].add(g)
    gene_to_pathways = defaultdict(set)
    for (pw, genes) in pathway_to_genes.items():
        for g in genes:
            gene_to_pathways[g].add(pw)
    return (pathway_to_genes, gene_to_pathways)

def _normalize_ncbi_gene_id(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == 'nan':
        return None
    if s.endswith('.0'):
        s = s[:-2]
    return s if s.isdigit() else None

def load_hgnc_ncbi_map(hgnc_file):
    if not hgnc_file or not os.path.exists(hgnc_file):
        return {}
    df = pd.read_csv(hgnc_file, sep='\t', low_memory=False)
    symbol_col = 'approved_symbol' if 'approved_symbol' in df.columns else 'symbol'
    entrez_col = next((c for c in ['entrez_id', 'entrez_gene_id', 'ncbi_gene_id'] if c in df.columns), None)
    if entrez_col is None:
        entrez_col = next((c for c in df.columns if 'entrez' in c.lower() or 'ncbi' in c.lower()), None)
    if entrez_col is None:
        return {}
    ncbi_to_symbol = {}
    for (_, row) in df.iterrows():
        gene_id = _normalize_ncbi_gene_id(row.get(entrez_col))
        symbol = normalize_symbol(row.get(symbol_col))
        if gene_id and symbol:
            ncbi_to_symbol[gene_id] = symbol
    return ncbi_to_symbol

def load_kegg_gene_sets(kegg_pathways_file, kegg_gene_pathway_file, hgnc_file):
    pathway_to_name = {}
    if kegg_pathways_file and os.path.exists(kegg_pathways_file):
        with open(kegg_pathways_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                pid = parts[0].split(':')[-1].strip()
                name = parts[1].strip()
                if pid and name:
                    pathway_to_name[pid] = name
    ncbi_to_symbol = load_hgnc_ncbi_map(hgnc_file)
    pathway_to_genes = defaultdict(set)
    if not kegg_gene_pathway_file or not os.path.exists(kegg_gene_pathway_file):
        return (pathway_to_genes, defaultdict(set))
    with open(kegg_gene_pathway_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = re.split('\\s+', line.strip())
            if len(parts) < 2:
                continue
            (a, b) = (parts[0], parts[1])
            gene_tok = a if 'hsa:' in a and 'path:' not in a else b
            path_tok = b if 'path:' in b or 'hsa0' in b else a
            gene_id = gene_tok.split(':')[-1].strip()
            pathway_id = path_tok.split(':')[-1].strip()
            symbol = ncbi_to_symbol.get(gene_id)
            if not symbol or not pathway_id:
                continue
            pathway_label = 'KEGG:' + pathway_to_name.get(pathway_id, pathway_id)
            pathway_to_genes[pathway_label].add(symbol)
    gene_to_pathways = defaultdict(set)
    for (pw, genes) in pathway_to_genes.items():
        for g in genes:
            gene_to_pathways[g].add(pw)
    return (pathway_to_genes, gene_to_pathways)

def load_uniprot_ensembl_protein_map(uniprot_file, alias_to_symbol):
    if not uniprot_file or not os.path.exists(uniprot_file):
        return defaultdict(set)
    df = pd.read_csv(uniprot_file, sep='\t', low_memory=False)
    gene_primary_col = next((c for c in ['Gene Names (primary)', 'Gene Names  (primary )'] if c in df.columns), None)
    gene_names_col = next((c for c in ['Gene Names', 'Gene names'] if c in df.columns), None)
    ensembl_cols = [c for c in df.columns if 'ensembl' in c.lower()]
    ensp_to_symbols = defaultdict(set)
    if not ensembl_cols:
        return ensp_to_symbols
    for (_, row) in df.iterrows():
        symbols = []
        if gene_primary_col:
            symbols.extend(split_symbol_field(row.get(gene_primary_col)))
        if gene_names_col:
            symbols.extend(split_symbol_field(row.get(gene_names_col)))
        symbols = [alias_to_symbol.get(s, s) for s in symbols]
        symbols = [s for s in symbols if s]
        if not symbols:
            continue
        joined = ' '.join((str(row.get(c, '')) for c in ensembl_cols))
        for ensp in re.findall('ENSP\\d+(?:\\.\\d+)?', joined, flags=re.IGNORECASE):
            ensp_to_symbols[ensp.upper().split('.')[0]].update(symbols)
    return ensp_to_symbols

def add_string_protein_graph(G_prot, string_file, ensp_to_symbols, gene_to_prot_idxs, weight=0.5, min_score=700, max_edges=2000000):
    if not string_file or not os.path.exists(string_file) or (not ensp_to_symbols):
        return 0
    opener = gzip.open if str(string_file).endswith('.gz') else open
    added = 0
    with opener(string_file, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except Exception:
                continue
            if score < float(min_score):
                continue
            p1 = parts[0].split('.')[-1].upper()
            p2 = parts[1].split('.')[-1].upper()
            genes1 = ensp_to_symbols.get(p1, set())
            genes2 = ensp_to_symbols.get(p2, set())
            if not genes1 or not genes2:
                continue
            w = float(weight) * float(score) / 1000.0
            for g1 in genes1:
                for g2 in genes2:
                    for i in gene_to_prot_idxs.get(g1, []):
                        for j in gene_to_prot_idxs.get(g2, []):
                            if i != j:
                                _add_symmetric_edge(G_prot, i, j, w)
                                added += 1
                                if added >= int(max_edges):
                                    return added
    return added

def load_cellmarker_prior(cellmarker_file, alias_to_symbol):
    if not cellmarker_file or not os.path.exists(cellmarker_file):
        return defaultdict(set)
    df = pd.read_excel(cellmarker_file) if cellmarker_file.endswith('.xlsx') else pd.read_csv(cellmarker_file, low_memory=False)
    cell_to_markers = defaultdict(set)
    cols = list(df.columns)
    col_map = {c.lower().replace('_', ''): c for c in cols}
    cell_col = col_map.get('cellname', col_map.get('celltype', None))
    marker_col = col_map.get('symbol', col_map.get('marker', None))
    if not cell_col or not marker_col:
        return cell_to_markers
    for (_, row) in df.iterrows():
        cell_type = str(row[cell_col]).strip().lower()
        symbol_str = str(row[marker_col])
        if cell_type == 'nan' or not cell_type or symbol_str == 'nan' or (not symbol_str):
            continue
        markers = re.split('[\\[\\]\\s,;]+', symbol_str)
        for m in markers:
            m = m.strip().upper()
            if m:
                cell_to_markers[cell_type].add(alias_to_symbol.get(m, m))
    return cell_to_markers

def load_proteinatlas_celltype_prior(proteinatlas_file, alias_to_symbol):
    if not proteinatlas_file or not os.path.exists(proteinatlas_file):
        return defaultdict(set)
    df = pd.read_csv(proteinatlas_file, sep='\t', low_memory=False)
    col_map = {c.lower().replace('_', '').replace(' ', ''): c for c in df.columns}
    gene_col = col_map.get('genename') or col_map.get('gene') or col_map.get('genesymbol') or next((c for c in df.columns if 'gene' in c.lower() and 'name' in c.lower()), None)
    cell_col = col_map.get('celltype') or col_map.get('singlecelltype') or col_map.get('cellline') or next((c for c in df.columns if 'cell' in c.lower() and 'type' in c.lower()), None)
    level_col = col_map.get('level') or col_map.get('expression') or col_map.get('reliability')
    if gene_col is None or cell_col is None:
        return defaultdict(set)
    accepted_levels = {'high', 'medium', 'enhanced', 'supported', 'approved', 'reliable'}
    cell_to_markers = defaultdict(set)
    for (_, row) in df.iterrows():
        gene = alias_to_symbol.get(normalize_symbol(row.get(gene_col)), normalize_symbol(row.get(gene_col)))
        cell_type = normalize_text(row.get(cell_col))
        if gene is None or cell_type is None:
            continue
        if level_col is not None:
            level = str(row.get(level_col, '')).strip().lower()
            if level and level not in accepted_levels and ('not detected' in level):
                continue
        cell_to_markers[cell_type.lower()].add(gene)
    return cell_to_markers

def _find_col(df, candidates, contains_any=None):
    norm = {c.lower().replace('_', '').replace(' ', ''): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace('_', '').replace(' ', '')
        if key in norm:
            return norm[key]
    if contains_any:
        for c in df.columns:
            low = c.lower()
            if all((tok in low for tok in contains_any)):
                return c
    return None

def _add_symmetric_edge(mat, i, j, w=1.0):
    if i is None or j is None or i == j:
        return
    mat[i, j] += float(w)
    mat[j, i] += float(w)

def add_dorothea_gene_graph(G_gene, dorothea_file, alias_to_symbol, gene_to_idx, weight=0.5):
    if not dorothea_file or not os.path.exists(dorothea_file):
        return 0
    df = pd.read_csv(dorothea_file, sep='\t', low_memory=False)
    src_col = _find_col(df, ['source', 'tf', 'transcription_factor', 'tf_symbol'], contains_any=['source'])
    tgt_col = _find_col(df, ['target', 'target_gene', 'target_symbol'], contains_any=['target'])
    conf_col = _find_col(df, ['confidence', 'level', 'dorothea_confidence'])
    if src_col is None or tgt_col is None:
        return 0
    conf_weight = {'a': 1.0, 'b': 0.8, 'c': 0.6, 'd': 0.4, 'e': 0.2}
    added = 0
    for (_, row) in df.iterrows():
        src = alias_to_symbol.get(normalize_symbol(row.get(src_col)), normalize_symbol(row.get(src_col)))
        tgt = alias_to_symbol.get(normalize_symbol(row.get(tgt_col)), normalize_symbol(row.get(tgt_col)))
        (i, j) = (gene_to_idx.get(src), gene_to_idx.get(tgt))
        if i is None or j is None or i == j:
            continue
        cw = 1.0
        if conf_col is not None:
            cw = conf_weight.get(str(row.get(conf_col, '')).strip().lower()[:1], 1.0)
        _add_symmetric_edge(G_gene, i, j, float(weight) * cw)
        added += 1
    return added

def add_omnipath_graphs(G_gene, G_prot, omnipath_file, alias_to_symbol, gene_to_idx, gene_to_prot_idxs, weight=0.5):
    if not omnipath_file or not os.path.exists(omnipath_file):
        return (0, 0)
    df = pd.read_csv(omnipath_file, sep='\t', low_memory=False)
    src_col = _find_col(df, ['source', 'genesymbol_intercell_source', 'source_genesymbol'])
    tgt_col = _find_col(df, ['target', 'genesymbol_intercell_target', 'target_genesymbol'])
    if src_col is None or tgt_col is None:
        return (0, 0)
    gene_edges = 0
    prot_edges = 0
    for (_, row) in df.iterrows():
        src = alias_to_symbol.get(normalize_symbol(row.get(src_col)), normalize_symbol(row.get(src_col)))
        tgt = alias_to_symbol.get(normalize_symbol(row.get(tgt_col)), normalize_symbol(row.get(tgt_col)))
        (gi, gj) = (gene_to_idx.get(src), gene_to_idx.get(tgt))
        if gi is not None and gj is not None and (gi != gj):
            _add_symmetric_edge(G_gene, gi, gj, weight)
            gene_edges += 1
        for pi in gene_to_prot_idxs.get(src, []):
            for pj in gene_to_prot_idxs.get(tgt, []):
                if pi != pj:
                    _add_symmetric_edge(G_prot, pi, pj, weight)
                    prot_edges += 1
    return (gene_edges, prot_edges)

def add_corum_protein_graph(G_prot, corum_file, alias_to_symbol, protein_to_idx, gene_to_prot_idxs, weight=0.5):
    if not corum_file or not os.path.exists(corum_file):
        return 0
    df = pd.read_csv(corum_file, sep='\t', low_memory=False)
    subunit_cols = [c for c in df.columns if 'subunit' in c.lower()]
    if not subunit_cols:
        return 0
    added = 0
    for (_, row) in df.iterrows():
        hit_idxs = set()
        for col in subunit_cols:
            for tok in split_symbol_field(row.get(col)):
                canon = alias_to_symbol.get(tok, tok)
                if canon in protein_to_idx:
                    hit_idxs.add(protein_to_idx[canon])
                hit_idxs.update(gene_to_prot_idxs.get(canon, []))
        hit_idxs = sorted(hit_idxs)
        for a in range(len(hit_idxs)):
            for b in range(a + 1, len(hit_idxs)):
                _add_symmetric_edge(G_prot, hit_idxs[a], hit_idxs[b], weight)
                added += 1
    return added

def build_multi_relation_kb_upgraded(protein_names, gene_names, hgnc_file, uniprot_file, reactome_uniprot_file, reactome_ensembl_file, cellmarker_file=None, dorothea_file=None, omnipath_file=None, corum_file=None, proteinatlas_file=None, kegg_pathways_file=None, kegg_gene_pathway_file=None, string_file=None, direct_weight=1.0, module_weight=0.35, celltype_weight=2.0, gene_graph_weight=1.0, prot_graph_weight=1.0, gg_celltype_weight=0.5, pp_celltype_weight=0.5, dorothea_graph_weight=0.5, omnipath_graph_weight=0.5, corum_graph_weight=0.5, proteinatlas_celltype_weight=0.5, kegg_pathway_weight=0.5, string_graph_weight=0.5):
    (alias_to_symbol, ensembl_to_symbol) = load_hgnc_maps(hgnc_file)
    (acc_to_symbols, alias_to_symbol, protein_name_to_accs) = load_uniprot_maps(uniprot_file, alias_to_symbol)
    (pathway_to_genes, gene_to_pathways) = load_reactome_gene_sets(reactome_uniprot_file, reactome_ensembl_file, acc_to_symbols, ensembl_to_symbol)
    (kegg_pathway_to_genes, kegg_gene_to_pathways) = load_kegg_gene_sets(kegg_pathways_file, kegg_gene_pathway_file, hgnc_file)
    if kegg_pathway_to_genes:
        for (pw, genes) in kegg_pathway_to_genes.items():
            pathway_to_genes[pw].update(genes)
        for (g, pathways) in kegg_gene_to_pathways.items():
            gene_to_pathways[g].update(pathways)
    cellmarker_cell_to_markers = load_cellmarker_prior(cellmarker_file, alias_to_symbol)
    proteinatlas_cell_to_markers = load_proteinatlas_celltype_prior(proteinatlas_file, alias_to_symbol)
    cell_to_markers = defaultdict(set)
    for (ct, markers) in cellmarker_cell_to_markers.items():
        cell_to_markers[ct].update(markers)
    if proteinatlas_cell_to_markers:
        for (ct, markers) in proteinatlas_cell_to_markers.items():
            cell_to_markers[ct].update(markers)
    protein_names_arr = np.asarray(protein_names).astype(str)
    gene_names_arr = np.asarray(gene_names).astype(str)
    (P, G) = (len(protein_names_arr), len(gene_names_arr))
    (gene_canon, gene_to_idx) = ([], {})
    for (i, g) in enumerate(gene_names_arr):
        gc = alias_to_symbol.get(normalize_symbol(g), normalize_symbol(g))
        gene_canon.append(gc)
        gene_to_idx[gc] = i
    (protein_canon, protein_to_gene_set) = ([], [])
    for p in protein_names_arr:
        pn = normalize_symbol(p)
        pc = alias_to_symbol.get(pn, pn)
        protein_canon.append(pc)
        gset = set()
        if pc in protein_name_to_accs:
            for acc in protein_name_to_accs[pc]:
                gset.update(acc_to_symbols.get(acc, set()))
        if pn in acc_to_symbols:
            gset.update(acc_to_symbols[pn])
        if pc in acc_to_symbols:
            gset.update(acc_to_symbols[pc])
        if pc in gene_to_idx:
            gset.add(pc)
        protein_to_gene_set.append(gset)
    protein_to_idx = {}
    gene_to_prot_idxs = defaultdict(list)
    for (p_idx, pc) in enumerate(protein_canon):
        protein_to_idx[pc] = p_idx
        for g in protein_to_gene_set[p_idx]:
            gene_to_prot_idxs[g].append(p_idx)
    M_pg_direct = np.zeros((P, G), dtype=np.float32)
    M_pg_module = np.zeros((P, G), dtype=np.float32)
    M_pg_celltype = np.zeros((P, G), dtype=np.float32)
    M_pg_proteinatlas = np.zeros((P, G), dtype=np.float32)
    for (p_idx, gset) in enumerate(protein_to_gene_set):
        for g in gset:
            if g in gene_to_idx:
                M_pg_direct[p_idx, gene_to_idx[g]] = 1.0
    for (p_idx, gset) in enumerate(protein_to_gene_set):
        pathways = set()
        for g in gset:
            pathways.update(gene_to_pathways.get(g, set()))
        gene_scores = defaultdict(float)
        for pw in pathways:
            genes_in_pw = [g for g in pathway_to_genes.get(pw, set()) if g in gene_to_idx]
            if not genes_in_pw:
                continue
            w_pw = 1.0 / np.sqrt(len(genes_in_pw))
            if str(pw).startswith('KEGG:'):
                w_pw *= float(kegg_pathway_weight)
            for g in genes_in_pw:
                gene_scores[g] += w_pw
        for (g, score) in gene_scores.items():
            M_pg_module[p_idx, gene_to_idx[g]] = score
    if cellmarker_cell_to_markers:
        for (p_idx, pc) in enumerate(protein_canon):
            relevant_cell_types = [ct for (ct, markers) in cellmarker_cell_to_markers.items() if pc in markers]
            for ct in relevant_cell_types:
                for g in cellmarker_cell_to_markers[ct]:
                    if g in gene_to_idx:
                        M_pg_celltype[p_idx, gene_to_idx[g]] += 1.0
    if proteinatlas_cell_to_markers:
        for (p_idx, pc) in enumerate(protein_canon):
            relevant_cell_types = [ct for (ct, markers) in proteinatlas_cell_to_markers.items() if pc in markers]
            for ct in relevant_cell_types:
                for g in proteinatlas_cell_to_markers[ct]:
                    if g in gene_to_idx:
                        M_pg_proteinatlas[p_idx, gene_to_idx[g]] += 1.0
    M_pg = direct_weight * M_pg_direct + module_weight * M_pg_module + celltype_weight * M_pg_celltype + proteinatlas_celltype_weight * M_pg_proteinatlas
    M_pg = normalize_matrix_rows(M_pg)
    M_gp = direct_weight * M_pg_direct.T + module_weight * M_pg_module.T + celltype_weight * M_pg_celltype.T + proteinatlas_celltype_weight * M_pg_proteinatlas.T
    M_gp = normalize_matrix_rows(M_gp)
    G_gene = np.zeros((G, G), dtype=np.float32)
    for i in range(G):
        pwi = gene_to_pathways.get(gene_canon[i], set())
        if not pwi:
            continue
        for j in range(i + 1, G):
            pwj = gene_to_pathways.get(gene_canon[j], set())
            inter = pwi.intersection(pwj)
            if inter:
                shared = sum((float(kegg_pathway_weight) if str(pw).startswith('KEGG:') else 1.0 for pw in inter))
                score = gene_graph_weight * shared / np.sqrt(max(len(pwi), 1) * max(len(pwj), 1))
                G_gene[i, j] += score
                G_gene[j, i] += score
    if cell_to_markers:
        gene_to_celltypes = defaultdict(set)
        for (ct, markers) in cell_to_markers.items():
            for g in markers:
                gene_to_celltypes[g].add(ct)
        for i in range(G):
            cti = gene_to_celltypes.get(gene_canon[i], set())
            if not cti:
                continue
            for j in range(i + 1, G):
                ctj = gene_to_celltypes.get(gene_canon[j], set())
                inter = cti.intersection(ctj)
                if inter:
                    score = gg_celltype_weight * len(inter) / np.sqrt(max(len(cti), 1) * max(len(ctj), 1))
                    G_gene[i, j] += score
                    G_gene[j, i] += score
    dorothea_gene_edges = add_dorothea_gene_graph(G_gene, dorothea_file=dorothea_file, alias_to_symbol=alias_to_symbol, gene_to_idx=gene_to_idx, weight=dorothea_graph_weight)
    G_prot = np.zeros((P, P), dtype=np.float32)
    for i in range(P):
        pwi = set()
        for g in protein_to_gene_set[i]:
            pwi.update(gene_to_pathways.get(g, set()))
        for j in range(i + 1, P):
            pwj = set()
            for g in protein_to_gene_set[j]:
                pwj.update(gene_to_pathways.get(g, set()))
            inter = pwi.intersection(pwj)
            if inter:
                shared = sum((float(kegg_pathway_weight) if str(pw).startswith('KEGG:') else 1.0 for pw in inter))
                score = prot_graph_weight * shared / np.sqrt(max(len(pwi), 1) * max(len(pwj), 1))
                G_prot[i, j] += score
                G_prot[j, i] += score
    if cell_to_markers:
        protein_to_celltypes = defaultdict(set)
        for pc in protein_canon:
            for (ct, markers) in cell_to_markers.items():
                if pc in markers:
                    protein_to_celltypes[pc].add(ct)
        for i in range(P):
            ci = protein_to_celltypes.get(protein_canon[i], set())
            if not ci:
                continue
            for j in range(i + 1, P):
                cj = protein_to_celltypes.get(protein_canon[j], set())
                inter = ci.intersection(cj)
                if inter:
                    score = pp_celltype_weight * len(inter) / np.sqrt(max(len(ci), 1) * max(len(cj), 1))
                    G_prot[i, j] += score
                    G_prot[j, i] += score
    (omnipath_gene_edges, omnipath_protein_edges) = add_omnipath_graphs(G_gene, G_prot, omnipath_file=omnipath_file, alias_to_symbol=alias_to_symbol, gene_to_idx=gene_to_idx, gene_to_prot_idxs=gene_to_prot_idxs, weight=omnipath_graph_weight)
    corum_protein_edges = add_corum_protein_graph(G_prot, corum_file=corum_file, alias_to_symbol=alias_to_symbol, protein_to_idx=protein_to_idx, gene_to_prot_idxs=gene_to_prot_idxs, weight=corum_graph_weight)
    ensp_to_symbols = load_uniprot_ensembl_protein_map(uniprot_file, alias_to_symbol)
    string_protein_edges = add_string_protein_graph(G_prot, string_file=string_file, ensp_to_symbols=ensp_to_symbols, gene_to_prot_idxs=gene_to_prot_idxs, weight=string_graph_weight, min_score=PROTEO_KB_STRING_MIN_SCORE, max_edges=PROTEO_KB_STRING_MAX_EDGES)
    G_gene = normalize_adjacency_with_selfloop(G_gene)
    G_prot = normalize_adjacency_with_selfloop(G_prot)
    stats = {'num_proteins': P, 'num_genes': G, 'resolved_proteins': int(np.sum(M_pg_direct.sum(axis=1) > 0)), 'num_direct_links': int((M_pg_direct > 0).sum()), 'num_module_links': int((M_pg_module > 0).sum()), 'num_celltype_links': int((M_pg_celltype > 0).sum()), 'num_proteinatlas_links': int((M_pg_proteinatlas > 0).sum()), 'gene_graph_edges': int(np.sum(G_gene > 0) - G), 'protein_graph_edges': int(np.sum(G_prot > 0) - P), 'dorothea_gene_edges': int(dorothea_gene_edges), 'omnipath_gene_edges': int(omnipath_gene_edges), 'omnipath_protein_edges': int(omnipath_protein_edges), 'corum_protein_edges': int(corum_protein_edges), 'string_protein_edges': int(string_protein_edges), 'proteinatlas_celltype_count': int(len(proteinatlas_cell_to_markers)), 'kegg_pathway_count': int(len(kegg_pathway_to_genes)), 'kegg_gene_memberships': int(sum((len(v) for v in kegg_pathway_to_genes.values())))}
    forced_genes = []
    for (g_idx, g) in enumerate(gene_names_arr):
        if any((m[:, g_idx].sum() > 0 for m in [M_pg_direct, M_pg_module, M_pg_celltype, M_pg_proteinatlas])):
            forced_genes.append(g)
    return (M_pg, M_gp, G_gene, G_prot, stats, forced_genes)

def adapt_proteo_builder_kwargs_with_agent_weights(builder_kwargs, selection, selected_sources, task_spec):
    builder_kwargs = dict(builder_kwargs or {})
    selection = dict(selection or {})
    agent_decision = selection.get('agent_decision', {}) if isinstance(selection.get('agent_decision', {}), dict) else {}
    raw_weights = agent_decision.get('source_weights', {}) if isinstance(agent_decision.get('source_weights', {}), dict) else {}
    raw_usage_modes = agent_decision.get('source_usage_modes', {}) if isinstance(agent_decision.get('source_usage_modes', {}), dict) else {}
    raw_source_actions = agent_decision.get('source_actions', {}) if isinstance(agent_decision.get('source_actions', {}), dict) else {}
    selected = [str(s).strip().lower() for s in selected_sources or [] if str(s).strip()]
    task_spec = dict(task_spec or {})
    source_modalities = {str(x).strip().lower() for x in task_spec.get('source_modalities', []) or [] if str(x).strip()}
    target_modality = str(task_spec.get('target_modality', '')).strip().lower()
    min_scale = float(os.getenv('KNOWLEDGE_AGENT_WEIGHT_MIN_SCALE', '0.70'))
    max_scale = float(os.getenv('KNOWLEDGE_AGENT_WEIGHT_MAX_SCALE', '1.25'))
    shrink = float(os.getenv('KNOWLEDGE_AGENT_WEIGHT_SHRINKAGE', '0.55'))
    numeric_weights = {}
    for src in selected:
        try:
            numeric_weights[src] = float(raw_weights.get(src, 0.0))
        except Exception:
            numeric_weights[src] = 0.0
    usage_modes = {src: str(raw_usage_modes.get(src, 'full')).strip().lower() or 'full' for src in selected}
    source_actions = {src: str(raw_source_actions.get(src, 'keep')).strip().lower() or 'keep' for src in selected}
    positive = [v for v in numeric_weights.values() if v > 0]
    if not positive:
        return (builder_kwargs, {'enabled': False, 'reason': 'agent_returned_no_positive_source_weights', 'source_weights': raw_weights})
    mean_w = float(np.mean(positive))

    def factor(*sources):
        vals = [numeric_weights.get(str(s).lower(), 0.0) for s in sources if str(s).lower() in selected]
        vals = [v for v in vals if v > 0]
        if not vals:
            return 1.0
        raw_scale = float(np.mean(vals)) / max(mean_w, 1e-08)
        stabilized = 1.0 + shrink * (raw_scale - 1.0)
        return float(np.clip(stabilized, min_scale, max_scale))

    def usage_multiplier(source_group: str, param: str) -> float:
        group_sources = [str(s).lower() for s in str(source_group).split('+')]
        active_modes = [usage_modes.get(s, 'full') for s in group_sources if s in usage_modes]
        active_actions = [source_actions.get(s, 'keep') for s in group_sources if s in source_actions]
        if not active_modes:
            return 1.0
        if any((m == 'disabled' for m in active_modes)):
            return 0.0
        if any((a == 'remove' for a in active_actions)):
            return 0.0
        structural_params = {'gene_graph_weight', 'prot_graph_weight', 'dorothea_graph_weight', 'omnipath_graph_weight', 'corum_graph_weight', 'kegg_pathway_weight', 'string_graph_weight'}
        prior_params = {'direct_weight', 'module_weight', 'celltype_weight', 'gg_celltype_weight', 'pp_celltype_weight', 'proteinatlas_celltype_weight'}
        scales = []
        for mode in active_modes:
            if mode == 'full':
                scales.append(1.0)
            elif mode == 'low_weight_full':
                scales.append(0.75)
            elif mode == 'graph_only':
                scales.append(0.95 if param in structural_params else 0.15)
            elif mode == 'prior_only':
                scales.append(0.95 if param in prior_params else 0.15)
            elif mode == 'warmup_only':
                scales.append(0.4)
            else:
                scales.append(1.0)
        scale = float(np.mean(scales))
        if any((a in {'downweight', 'warmup_only'} for a in active_actions)):
            scale *= 0.85
        return float(np.clip(scale, 0.0, max_scale))

    def template_multiplier(param: str) -> float:
        if target_modality == 'gene' and 'protein' not in source_modalities:
            conservative = {'direct_weight': 1.01, 'module_weight': 1.0, 'gene_graph_weight': 1.0, 'prot_graph_weight': 1.0, 'celltype_weight': 1.0, 'gg_celltype_weight': 1.0, 'pp_celltype_weight': 1.0, 'kegg_pathway_weight': 1.0, 'proteinatlas_celltype_weight': 1.0}
            return conservative.get(param, 1.0)
        if target_modality == 'protein' and 'gene' not in source_modalities:
            hints = {'direct_weight': 1.01, 'celltype_weight': 1.02, 'gg_celltype_weight': 1.02, 'pp_celltype_weight': 1.02, 'proteinatlas_celltype_weight': 1.04 if 'proteinatlas' in selected else 1.0, 'kegg_pathway_weight': 1.0}
            return hints.get(param, 1.0)
        if target_modality == 'gene' and 'protein' in source_modalities:
            hints = {'direct_weight': 1.02, 'celltype_weight': 1.0, 'gg_celltype_weight': 1.0, 'pp_celltype_weight': 1.0, 'proteinatlas_celltype_weight': 1.02 if 'proteinatlas' in selected else 1.0, 'module_weight': 1.0, 'gene_graph_weight': 1.0, 'prot_graph_weight': 1.0}
            return hints.get(param, 1.0)
        if target_modality == 'protein' and 'gene' in source_modalities:
            hints = {'direct_weight': 1.03, 'celltype_weight': 1.01, 'gg_celltype_weight': 1.01, 'pp_celltype_weight': 1.01, 'proteinatlas_celltype_weight': 1.06 if 'proteinatlas' in selected else 1.0, 'kegg_pathway_weight': 1.03 if 'kegg' in selected else 1.0}
            return hints.get(param, 1.0)
        return 1.0
    source_to_param = {'uniprot+hgnc': ('direct_weight', factor('uniprot', 'hgnc')), 'reactome': ('module_weight', factor('reactome')), 'reactome_gene_graph': ('gene_graph_weight', factor('reactome')), 'reactome_protein_graph': ('prot_graph_weight', factor('reactome')), 'cellmarker': ('celltype_weight', factor('cellmarker')), 'cellmarker_gene_graph': ('gg_celltype_weight', factor('cellmarker')), 'cellmarker_protein_graph': ('pp_celltype_weight', factor('cellmarker')), 'proteinatlas': ('proteinatlas_celltype_weight', factor('proteinatlas')), 'dorothea': ('dorothea_graph_weight', factor('dorothea')), 'omnipath': ('omnipath_graph_weight', factor('omnipath')), 'corum': ('corum_graph_weight', factor('corum')), 'kegg': ('kegg_pathway_weight', factor('kegg')), 'string': ('string_graph_weight', factor('string'))}
    applied = {}
    for (label, (param, scale)) in source_to_param.items():
        if param not in builder_kwargs:
            continue
        old = float(builder_kwargs[param])
        mode_scale = usage_multiplier(label, param)
        adjusted_scale = float(np.clip(float(scale) * template_multiplier(param) * mode_scale, 0.0, max_scale))
        new = old * adjusted_scale
        builder_kwargs[param] = new
        applied[param] = {'source_group': label, 'base': old, 'scale': adjusted_scale, 'mode_scale': mode_scale, 'effective': new}
    return (builder_kwargs, {'enabled': True, 'task_spec': task_spec, 'selected_sources': selected, 'source_weights': numeric_weights, 'source_usage_modes': usage_modes, 'source_actions': source_actions, 'mean_positive_weight': mean_w, 'applied': applied})
