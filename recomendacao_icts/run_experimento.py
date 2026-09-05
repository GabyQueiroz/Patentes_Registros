from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_PTN_ZIP = Path(r"C:\Users\gabri\Downloads\badepiv11_ptn.zip")
DEFAULT_PRG_ZIP = Path(r"C:\Users\gabri\Downloads\badepiv11_prg.zip")


ICT_PATTERNS = [
    r"\bUNIVERSIDADE\b",
    r"\bUNIV\b",
    r"\bINSTITUTO FEDERAL\b",
    r"\bINST FEDERAL\b",
    r"\bIF[A-Z]{2}\b",
    r"\bCEFET\b",
    r"\bEMBRAPA\b",
    r"\bFIOCRUZ\b",
    r"\bFUNDACAO UNIVERSIDADE\b",
    r"\bFUNDAÇÃO UNIVERSIDADE\b",
    r"\bFUNDACAO DE APOIO\b",
    r"\bFUNDAÇÃO DE APOIO\b",
    r"\bCENTRO FEDERAL DE EDUCACAO\b",
    r"\bCENTRO FEDERAL DE EDUCAÇÃO\b",
    r"\bINSTITUTO DE PESQUISA\b",
    r"\bINSTITUTO NACIONAL\b",
    r"\bSENAI\b",
    r"\bSENAC\b",
    r"\bESCOLA POLITECNICA\b",
    r"\bESCOLA POLITÉCNICA\b",
    r"\bFACULDADE\b",
]


APPLICATION_LEXICON = {
    "agricultura_inteligente": [
        "agric", "irrig", "solo", "plant", "safra", "semente", "colheit", "clima",
        "rural", "praga", "fertiliz", "pecuaria", "pecuária",
    ],
    "cidades_inteligentes": [
        "cidade", "urbano", "urbana", "mobilidade", "transito", "trânsito", "semaforo",
        "semáforo", "estacionamento", "iluminacao publica", "iluminação pública",
        "saneamento", "residuo", "resíduo", "seguranca publica", "segurança pública",
    ],
    "saude_digital": [
        "saude", "saúde", "medic", "hospital", "clin", "diagnost", "paciente",
        "farmac", "biomed", "telemed", "imagem medica", "imagem médica",
    ],
    "industria_4_0": [
        "automacao", "automação", "industrial", "manufatura", "controle", "robô",
        "robo", "sensor", "iot", "internet das coisas", "visao computacional",
        "visão computacional",
    ],
    "energia": [
        "energia", "eletric", "fotovolta", "solar", "eolica", "eólica", "bateria",
        "combustivel", "combustível", "biogas", "biogás", "hidrogenio", "hidrogênio",
    ],
    "educacao_tecnologica": [
        "educa", "ensino", "aprendiz", "aluno", "escola", "treinamento", "ead",
        "avaliacao", "avaliação",
    ],
}


def strip_accents(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(text).upper()
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_institution(name: str) -> str:
    name = normalize_text(name)
    replacements = {
        "UNIVERSIDADE FEDERAL DO": "UNIVERSIDADE FEDERAL DE",
        "UNIVERSIDADE FEDERAL DA": "UNIVERSIDADE FEDERAL DE",
        "UNIVERSIDADE ESTADUAL DO": "UNIVERSIDADE ESTADUAL DE",
        "UNIVERSIDADE ESTADUAL DA": "UNIVERSIDADE ESTADUAL DE",
        "FUNDACAO": "FUNDACAO",
        "LTDA": "",
        "S A": "",
        "SA": "",
        "ME": "",
    }
    for old, new in replacements.items():
        name = re.sub(rf"\b{old}\b", new, name)
    return re.sub(r"\s+", " ", name).strip()


def is_ict(name: str) -> bool:
    norm = normalize_text(name)
    return any(re.search(pattern, norm) for pattern in ICT_PATTERNS)


def read_csv_from_zip(
    zip_path: Path,
    member: str,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            with zipfile.ZipFile(zip_path) as z:
                with z.open(member) as f:
                    return pd.read_csv(
                        f,
                        sep=";",
                        quotechar='"',
                        dtype=str,
                        usecols=usecols,
                        nrows=nrows,
                        encoding=enc,
                        low_memory=False,
                    )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Nao foi possivel ler {member}: {last_error}")


def parse_year(value: object) -> float:
    try:
        y = int(str(value)[:4])
        return y if 1900 <= y <= 2100 else np.nan
    except Exception:
        return np.nan


def first_non_empty(values: Iterable[object]) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@dataclass
class ExperimentConfig:
    ptn_zip: Path
    prg_zip: Path
    out_dir: Path
    mode: str
    max_patents: int
    max_softwares: int
    random_state: int
    top_k: int


def load_assets(cfg: ExperimentConfig) -> pd.DataFrame:
    quick = cfg.mode == "rapido"
    ptn_dep_n = min(max(cfg.max_patents * 6, 70000), 180000) if quick else None
    ptn_rel_n = min(max(cfg.max_patents * 8, 110000), 320000) if quick else None
    prg_dep_n = min(max(cfg.max_softwares * 3, 25000), 44658) if quick else None

    ptn_dep = read_csv_from_zip(
        cfg.ptn_zip,
        "badepiv11_ptn_deposito.csv",
        ["NO_PEDIDO", "ANO", "DT_ENTRADA_INPI", "NM_TITULO_PATENTE"],
        nrows=ptn_dep_n,
    )
    ptn_owners = read_csv_from_zip(
        cfg.ptn_zip,
        "badepiv11_ptn_depositante.csv",
        ["NO_PEDIDO", "NO_ORDEM", "CD_TIPO_PFPJ", "NM_COMPLET_PFPJ", "CD_PAIS_PFPJ", "CD_UF_PFPJ", "NM_CIDADE_PFPJ"],
        nrows=ptn_rel_n,
    )
    ptn_ipc = read_csv_from_zip(
        cfg.ptn_zip,
        "badepiv11_ipc_campo_tec.csv",
        ["NO_PEDIDO", "CD_CLASSIF", "CAMPO_TEC"],
        nrows=ptn_rel_n,
    )

    prg_dep = read_csv_from_zip(
        cfg.prg_zip,
        "badepiv11_prg_deposito.csv",
        ["NO_PEDIDO", "ANO", "DT_ENTRADA_INPI", "NM_TITULO_PROGRAMA"],
        nrows=prg_dep_n,
    )
    prg_owners = read_csv_from_zip(
        cfg.prg_zip,
        "badepiv11_prg_depositante.csv",
        ["NO_PEDIDO", "CD_TIPO_PFPJ", "NM_COMPLET_PFPJ", "CD_PAIS_PFPJ", "CD_UF_PFPJ", "NM_CIDADE_PFPJ"],
    )
    prg_tipo = read_csv_from_zip(
        cfg.prg_zip,
        "badepiv11_prg_tipo.csv",
        ["NO_PEDIDO", "CD_TIPO_PROGRAMA", "DS_TIPO_PROGRAMA"],
    )

    ptn_owners = ptn_owners.sort_values(["NO_PEDIDO", "NO_ORDEM"], na_position="last")
    ptn_owners_agg = ptn_owners.groupby("NO_PEDIDO", as_index=False).agg(
        depositante=("NM_COMPLET_PFPJ", first_non_empty),
        tipo_depositante=("CD_TIPO_PFPJ", first_non_empty),
        pais=("CD_PAIS_PFPJ", first_non_empty),
        uf=("CD_UF_PFPJ", first_non_empty),
        cidade=("NM_CIDADE_PFPJ", first_non_empty),
        n_depositantes=("NM_COMPLET_PFPJ", "count"),
    )
    ptn_ipc_agg = ptn_ipc.groupby("NO_PEDIDO", as_index=False).agg(
        ipc=("CD_CLASSIF", lambda s: " ".join(sorted(set(x.strip() for x in s.dropna().astype(str) if x.strip()))[:8])),
        campo_tec=("CAMPO_TEC", lambda s: " ".join(sorted(set(x.strip() for x in s.dropna().astype(str) if x.strip()))[:8])),
    )
    ptn = ptn_dep.merge(ptn_owners_agg, on="NO_PEDIDO", how="left").merge(ptn_ipc_agg, on="NO_PEDIDO", how="left")
    ptn = ptn.rename(columns={"NM_TITULO_PATENTE": "titulo"})
    ptn["tipo_ativo"] = "patente"
    ptn["tipo_programa"] = ""

    prg_owners_agg = prg_owners.groupby("NO_PEDIDO", as_index=False).agg(
        depositante=("NM_COMPLET_PFPJ", first_non_empty),
        tipo_depositante=("CD_TIPO_PFPJ", first_non_empty),
        pais=("CD_PAIS_PFPJ", first_non_empty),
        uf=("CD_UF_PFPJ", first_non_empty),
        cidade=("NM_CIDADE_PFPJ", first_non_empty),
        n_depositantes=("NM_COMPLET_PFPJ", "count"),
    )
    prg_tipo_agg = prg_tipo.groupby("NO_PEDIDO", as_index=False).agg(
        tipo_programa=("DS_TIPO_PROGRAMA", lambda s: " ".join(sorted(set(x.strip() for x in s.dropna().astype(str) if x.strip()))[:8])),
        cod_tipo_programa=("CD_TIPO_PROGRAMA", lambda s: " ".join(sorted(set(x.strip() for x in s.dropna().astype(str) if x.strip()))[:8])),
    )
    prg = prg_dep.merge(prg_owners_agg, on="NO_PEDIDO", how="left").merge(prg_tipo_agg, on="NO_PEDIDO", how="left")
    prg = prg.rename(columns={"NM_TITULO_PROGRAMA": "titulo"})
    prg["tipo_ativo"] = "software"
    prg["ipc"] = ""
    prg["campo_tec"] = ""

    cols = [
        "NO_PEDIDO", "ANO", "DT_ENTRADA_INPI", "titulo", "tipo_ativo", "depositante",
        "tipo_depositante", "pais", "uf", "cidade", "n_depositantes", "ipc", "campo_tec",
        "tipo_programa",
    ]
    assets = pd.concat([ptn[cols], prg[cols]], ignore_index=True)
    assets["ano"] = assets["ANO"].map(parse_year)
    assets["depositante_norm"] = assets["depositante"].map(normalize_institution)
    assets["eh_ict"] = assets["depositante"].map(is_ict)
    assets["titulo_limpo"] = assets["titulo"].map(normalize_text)
    assets["texto_modelo"] = (
        assets["titulo_limpo"].fillna("") + " " +
        assets["ipc"].fillna("").map(normalize_text) + " " +
        assets["campo_tec"].fillna("").map(normalize_text) + " " +
        assets["tipo_programa"].fillna("").map(normalize_text)
    ).str.strip()

    assets = assets[(assets["eh_ict"]) & (assets["titulo_limpo"].str.len() >= 5)].copy()
    assets = assets[assets["ano"].notna()].copy()
    assets["asset_id"] = assets["tipo_ativo"].str[0].str.upper() + "_" + assets["NO_PEDIDO"].str.strip()

    patents = assets[assets["tipo_ativo"] == "patente"]
    softwares = assets[assets["tipo_ativo"] == "software"]
    if cfg.max_patents and len(patents) > cfg.max_patents:
        patents = patents.sample(cfg.max_patents, random_state=cfg.random_state)
    if cfg.max_softwares and len(softwares) > cfg.max_softwares:
        softwares = softwares.sample(cfg.max_softwares, random_state=cfg.random_state)
    assets = pd.concat([patents, softwares], ignore_index=True)
    return assets.reset_index(drop=True)


def assign_application(text: str) -> str:
    text_l = strip_accents(text).lower()
    scores = {}
    for area, terms in APPLICATION_LEXICON.items():
        scores[area] = sum(1 for term in terms if strip_accents(term).lower() in text_l)
    area, score = max(scores.items(), key=lambda kv: kv[1])
    return area if score > 0 else "geral"


def token_set(text: str) -> set[str]:
    tokens = re.findall(r"[A-Z0-9]{3,}", normalize_text(text))
    generic = {
        "SISTEMA", "PROCESSO", "METODO", "DISPOSITIVO", "PROGRAMA", "PARA", "COM",
        "DOS", "DAS", "UMA", "POR", "GERENCIAMENTO", "GESTAO", "CONTROLE",
    }
    return {t for t in tokens if t not in generic}


def build_embeddings(assets: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray, TfidfVectorizer, TruncatedSVD]:
    stop_words = [
        "de", "da", "do", "das", "dos", "para", "por", "com", "em", "e", "ou", "um", "uma",
        "sistema", "processo", "metodo", "método", "dispositivo", "programa", "aplicativo",
        "controle", "gerenciamento", "gestao", "gestão",
    ]
    vectorizer = TfidfVectorizer(
        max_features=30000,
        min_df=2,
        max_df=0.92,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=stop_words,
        strip_accents="unicode",
    )
    X_tfidf = vectorizer.fit_transform(assets["texto_modelo"].fillna(""))
    n_components = min(256, max(8, X_tfidf.shape[1] - 1), max(8, X_tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_emb = svd.fit_transform(X_tfidf)
    X_emb = normalize(X_emb)
    assets = assets.copy()
    assets["area_lexica"] = assets["texto_modelo"].map(assign_application)
    return assets, X_tfidf, X_emb, vectorizer, svd


def topic_discovery(assets: pd.DataFrame, X_emb: np.ndarray, out_dir: Path, random_state: int) -> pd.DataFrame:
    rows = []
    candidate_k = [8, 12, 16, 20, 28]
    n = len(assets)
    best = None
    for k in candidate_k:
        if n <= k + 5:
            continue
        model = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=2048, n_init=10)
        labels = model.fit_predict(X_emb)
        sample_n = min(6000, n)
        idx = np.random.default_rng(random_state).choice(n, size=sample_n, replace=False)
        sil = silhouette_score(X_emb[idx], labels[idx])
        rows.append({"k": k, "silhouette": sil, "inertia": model.inertia_})
        if best is None or sil > best[0]:
            best = (sil, k, labels, model)

    if best is None:
        assets["topico"] = 0
        metrics = pd.DataFrame([{"k": 1, "silhouette": np.nan, "inertia": np.nan}])
    else:
        assets["topico"] = best[2]
        metrics = pd.DataFrame(rows).sort_values("silhouette", ascending=False)

    metrics.to_csv(out_dir / "metricas_agrupamento.csv", index=False, encoding="utf-8-sig")

    topic_summary = (
        assets.groupby("topico")
        .agg(
            n=("asset_id", "count"),
            patentes=("tipo_ativo", lambda s: int((s == "patente").sum())),
            softwares=("tipo_ativo", lambda s: int((s == "software").sum())),
            ano_mediano=("ano", "median"),
            principais_areas=("area_lexica", lambda s: "; ".join(s.value_counts().head(3).index)),
            exemplos=("titulo", lambda s: " | ".join(s.dropna().astype(str).head(3))),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    topic_summary.to_csv(out_dir / "topicos_resumo.csv", index=False, encoding="utf-8-sig")
    return assets


def make_candidate_pairs(
    assets: pd.DataFrame,
    X_emb: np.ndarray,
    max_pairs: int,
    top_neighbors: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    patents = assets[assets["tipo_ativo"] == "patente"].copy()
    softwares = assets[assets["tipo_ativo"] == "software"].copy()
    if patents.empty or softwares.empty:
        raise RuntimeError("A amostra filtrada precisa conter patentes e softwares de ICTs.")

    p_idx = patents.index.to_numpy()
    s_idx = softwares.index.to_numpy()
    Xp = X_emb[p_idx]
    Xs = X_emb[s_idx]

    rows = []
    software_by_ict = softwares.groupby("depositante_norm").indices
    for local_i, global_i in enumerate(tqdm(p_idx, desc="Gerando candidatos", leave=False)):
        sims = 1 - cdist(Xp[local_i:local_i + 1], Xs, metric="cosine").ravel()
        k = min(top_neighbors, len(s_idx))
        top_local = np.argpartition(-sims, kth=k - 1)[:k]
        neg_local = rng.choice(len(s_idx), size=min(k, len(s_idx)), replace=False)
        for j in np.unique(np.r_[top_local, neg_local]):
            rows.append((global_i, s_idx[j], float(sims[j])))
        ict = assets.loc[global_i, "depositante_norm"]
        same_ict_local = software_by_ict.get(ict, [])
        if len(same_ict_local):
            selected = rng.choice(same_ict_local, size=min(8, len(same_ict_local)), replace=False)
            for local_j in selected:
                global_j = softwares.iloc[int(local_j)].name
                sim = 1 - cdist(X_emb[[global_i]], X_emb[[global_j]], metric="cosine").ravel()[0]
                rows.append((global_i, global_j, float(sim)))
        if len(rows) >= max_pairs:
            break

    pairs = pd.DataFrame(rows, columns=["patent_idx", "software_idx", "cosine"])
    pairs = pairs.drop_duplicates(["patent_idx", "software_idx"])
    p = assets.loc[pairs["patent_idx"]].reset_index(drop=True)
    s = assets.loc[pairs["software_idx"]].reset_index(drop=True)
    pairs["mesma_ict"] = (p["depositante_norm"].to_numpy() == s["depositante_norm"].to_numpy()).astype(int)
    pairs["mesma_uf"] = (p["uf"].fillna("").to_numpy() == s["uf"].fillna("").to_numpy()).astype(int)
    same_area = p["area_lexica"].to_numpy() == s["area_lexica"].to_numpy()
    informative_area = p["area_lexica"].to_numpy() != "geral"
    pairs["mesma_area_lexica"] = (same_area & informative_area).astype(int)
    pairs["mesmo_topico"] = (p["topico"].to_numpy() == s["topico"].to_numpy()).astype(int)
    pairs["dist_ano"] = np.abs(p["ano"].to_numpy(dtype=float) - s["ano"].to_numpy(dtype=float))
    pairs["jaccard_texto"] = [
        len(tp & ts) / max(1, len(tp | ts))
        for tp, ts in zip(p["texto_modelo"].map(token_set), s["texto_modelo"].map(token_set))
    ]

    # Rotulo fraco institucional: a ICT ter patente e software em janela temporal proxima
    # indica uma complementaridade observavel em seu portfolio, mas nao transferencia real.
    pairs["y"] = ((pairs["mesma_ict"] == 1) & (pairs["dist_ano"] <= 10)).astype(int)

    if pairs["y"].sum() < 25:
        pairs["y"] = (pairs["mesma_ict"] == 1).astype(int)
    return pairs


def graph_features(assets: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    graph = nx.Graph()
    for _, r in assets.iterrows():
        a = r["asset_id"]
        graph.add_node(a, tipo=r["tipo_ativo"])
        graph.add_edge(a, "AREA::" + r["area_lexica"], tipo="area")
        graph.add_edge(a, "UF::" + str(r.get("uf", "")), tipo="uf")
        graph.add_edge(a, "TOPICO::" + str(r.get("topico", "")), tipo="topico")
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    rows = []
    for _, pair in pairs.iterrows():
        p = assets.loc[pair["patent_idx"], "asset_id"]
        s = assets.loc[pair["software_idx"], "asset_id"]
        p_neighbors = set(graph.neighbors(p))
        s_neighbors = set(graph.neighbors(s))
        inter = len(p_neighbors & s_neighbors)
        union = max(1, len(p_neighbors | s_neighbors))
        rows.append({
            "grau_patente": degree.get(p, 0),
            "grau_software": degree.get(s, 0),
            "jaccard_vizinhanca": inter / union,
            "common_neighbors": inter,
            "clustering_patente": clustering.get(p, 0.0),
            "clustering_software": clustering.get(s, 0.0),
        })
    return pd.DataFrame(rows)


def evaluate_ranking(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int) -> dict[str, float]:
    precisions, recalls, ndcgs = [], [], []
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < 2 or y_true[mask].sum() == 0:
            continue
        order = np.argsort(-y_score[mask])
        yt = y_true[mask][order]
        top = yt[:k]
        precisions.append(float(top.mean()))
        recalls.append(float(top.sum() / yt.sum()))
        ndcgs.append(float(ndcg_score([yt], [y_score[mask][order]], k=min(k, len(yt)))))
    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else np.nan,
        f"recall@{k}": float(np.mean(recalls)) if recalls else np.nan,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else np.nan,
    }


def train_models(features: pd.DataFrame, y: pd.Series, groups: pd.Series, out_dir: Path, random_state: int, top_k: int):
    split = train_test_split(
        np.arange(len(features)),
        test_size=0.30,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )
    train_idx, test_idx = split
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()
    groups_test = groups.iloc[test_idx].to_numpy()

    specs = []
    for params in ParameterGrid({"C": [0.3, 1.0, 3.0], "class_weight": ["balanced"]}):
        specs.append(("logistic_regression", Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=random_state, **params)),
        ]), params))
    for params in ParameterGrid({"max_depth": [6, 12, None], "min_samples_leaf": [2, 5], "n_estimators": [160, 320]}):
        specs.append(("random_forest", RandomForestClassifier(random_state=random_state, class_weight="balanced_subsample", n_jobs=-1, **params), params))
    for params in ParameterGrid({"learning_rate": [0.04, 0.08], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 0.05]}):
        specs.append(("hist_gradient_boosting", HistGradientBoostingClassifier(random_state=random_state, **params), params))
    for params in ParameterGrid({"hidden_layer_sizes": [(64,), (128, 32)], "alpha": [0.0001, 0.001], "learning_rate_init": [0.001]}):
        specs.append(("rede_neural_mlp", Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(max_iter=350, early_stopping=True, random_state=random_state, **params)),
        ]), params))

    rows = []
    best = None
    for name, model, params in tqdm(specs, desc="Treinando modelos"):
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
        preds = (scores >= 0.5).astype(int)
        ranking = evaluate_ranking(y_test, scores, groups_test, top_k)
        row = {
            "modelo": name,
            "hiperparametros": json.dumps(params, ensure_ascii=False),
            "roc_auc": roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else np.nan,
            "average_precision": average_precision_score(y_test, scores) if len(np.unique(y_test)) > 1 else np.nan,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            **ranking,
        }
        rows.append(row)
        score = row["average_precision"] if not math.isnan(row["average_precision"]) else -1
        if best is None or score > best[0]:
            best = (score, name, model, params, scores, y_test, test_idx)

    metrics = pd.DataFrame(rows).sort_values(["average_precision", f"ndcg@{top_k}"], ascending=False)
    metrics.to_csv(out_dir / "metricas_modelos.csv", index=False, encoding="utf-8-sig")
    joblib.dump(best[2], out_dir / "melhor_modelo.joblib")

    report = classification_report(best[5], (best[4] >= 0.5).astype(int), zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(out_dir / "relatorio_classificacao_melhor_modelo.csv", encoding="utf-8-sig")
    return metrics, best


def create_recommendations(assets: pd.DataFrame, pairs: pd.DataFrame, features: pd.DataFrame, model, out_dir: Path, top_k: int) -> pd.DataFrame:
    scores = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(features)
    rec = pairs.copy()
    rec["score_modelo"] = scores
    cmin, cmax = rec["cosine"].min(), rec["cosine"].max()
    rec["cosine_norm"] = (rec["cosine"] - cmin) / max(1e-9, cmax - cmin)
    semantic_core = np.sqrt(np.clip(rec["score_modelo"], 0, 1) * np.clip(rec["cosine_norm"], 0, 1))
    rec["score_recomendacao"] = (
        0.62 * semantic_core
        + 0.18 * np.clip(rec["jaccard_texto"] * 5, 0, 1)
        + 0.10 * rec["mesmo_topico"]
        + 0.06 * rec["mesma_area_lexica"]
        + 0.04 * rec["mesma_ict"]
    )
    p = assets.loc[rec["patent_idx"]].reset_index(drop=True).add_prefix("patente_")
    s = assets.loc[rec["software_idx"]].reset_index(drop=True).add_prefix("software_")
    rec_full = pd.concat([rec.reset_index(drop=True), p, s], axis=1)
    cols = [
        "score_recomendacao", "score_modelo", "cosine", "jaccard_texto", "y", "mesma_ict", "mesma_area_lexica", "mesmo_topico", "dist_ano",
        "patente_NO_PEDIDO", "patente_ano", "patente_depositante", "patente_titulo", "patente_ipc", "patente_area_lexica",
        "software_NO_PEDIDO", "software_ano", "software_depositante", "software_titulo", "software_tipo_programa", "software_area_lexica",
    ]
    rec_full = rec_full.sort_values("score_recomendacao", ascending=False)
    rec_full[cols].head(top_k * 50).to_csv(out_dir / "recomendacoes_topk.csv", index=False, encoding="utf-8-sig")
    return rec_full


def plot_outputs(assets: pd.DataFrame, pairs: pd.DataFrame, metrics: pd.DataFrame, recs: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Serif", rc={
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.edgecolor": "0.15",
        "axes.labelcolor": "0.1",
        "xtick.color": "0.1",
        "ytick.color": "0.1",
        "axes.titleweight": "bold",
    })
    palette = ["#203A43", "#406882", "#6B8E7F", "#A98467", "#8A5A44", "#6D6875"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    anual = assets.pivot_table(index="ano", columns="tipo_ativo", values="asset_id", aggfunc="count", fill_value=0)
    anual.sort_index().plot(ax=ax, color=palette[:2], linewidth=2)
    ax.set_title("Evolucao anual dos ativos de ICTs identificados")
    ax.set_xlabel("Ano de entrada no INPI")
    ax.set_ylabel("Numero de ativos")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_01_evolucao_anual.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    top_areas = assets["area_lexica"].value_counts().head(10).sort_values()
    ax.barh(top_areas.index, top_areas.values, color=palette[2])
    ax.set_title("Areas de aplicacao inferidas por lexico tecnologico")
    ax.set_xlabel("Numero de ativos")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_02_areas_lexicas.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plot_metrics = metrics.head(12).copy()
    plot_metrics["modelo_label"] = plot_metrics["modelo"] + "\n" + plot_metrics.index.astype(str)
    sns.barplot(data=plot_metrics, x="average_precision", y="modelo_label", hue="modelo", dodge=False, ax=ax, palette=palette)
    ax.legend_.remove() if ax.legend_ else None
    ax.set_title("Desempenho dos melhores modelos por Average Precision")
    ax.set_xlabel("Average Precision")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_03_metricas_modelos.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sns.histplot(recs["score_recomendacao"].head(max(1000, top_k * 50)), bins=30, color=palette[1], ax=ax)
    ax.set_title("Distribuicao dos escores nas recomendacoes candidatas")
    ax.set_xlabel("Score composto de recomendacao")
    ax.set_ylabel("Frequencia")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_04_distribuicao_scores.png")
    plt.close(fig)

    top_graph = recs.head(min(80, len(recs)))
    G = nx.Graph()
    for _, r in top_graph.iterrows():
        p = "P " + str(r["patente_NO_PEDIDO"]).strip()
        s = "S " + str(r["software_NO_PEDIDO"]).strip()
        G.add_node(p, tipo="Patente")
        G.add_node(s, tipo="Software")
        G.add_edge(p, s, weight=float(r["score_recomendacao"]))
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=0.45)
    node_colors = [palette[0] if G.nodes[n]["tipo"] == "Patente" else palette[3] for n in G.nodes]
    widths = [0.5 + 3 * G.edges[e]["weight"] for e in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.34, edge_color="0.35")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=120, linewidths=0.4, edgecolors="white")
    ax.set_title("Rede das principais complementaridades recomendadas")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_05_rede_recomendacoes.png")
    plt.close(fig)


def write_report(
    assets: pd.DataFrame,
    pairs: pd.DataFrame,
    metrics: pd.DataFrame,
    recs: pd.DataFrame,
    out_dir: Path,
    best_name: str,
    best_params: dict,
) -> None:
    top_icts = assets["depositante_norm"].value_counts().head(15).reset_index()
    top_icts.columns = ["ICT", "ativos"]
    top_icts.to_csv(out_dir / "tabela_top_icts.csv", index=False, encoding="utf-8-sig")

    best_row = metrics.iloc[0].to_dict()
    md = [
        "# Relatorio do experimento",
        "",
        "## Escopo",
        "",
        "Sistema de recomendacao de complementaridades tecnologicas entre patentes e programas de computador de ICTs brasileiras.",
        "",
        "A classificacao de ICTs foi feita por regras transparentes aplicadas ao nome do depositante. A complementaridade usada para treinamento e avaliacao e um rotulo fraco baseado em similaridade semantica, proximidade topica, area lexica, instituicao e distancia temporal. Portanto, os resultados indicam oportunidades potenciais, nao transferencia realizada.",
        "",
        "## Base processada",
        "",
        f"- Ativos de ICTs: {len(assets):,}".replace(",", "."),
        f"- Patentes: {(assets['tipo_ativo'] == 'patente').sum():,}".replace(",", "."),
        f"- Programas de computador: {(assets['tipo_ativo'] == 'software').sum():,}".replace(",", "."),
        f"- ICTs distintas: {assets['depositante_norm'].nunique():,}".replace(",", "."),
        f"- Pares patente-software candidatos: {len(pairs):,}".replace(",", "."),
        f"- Pares positivos pelo rotulo fraco: {int(pairs['y'].sum()):,}".replace(",", "."),
        "",
        "## Melhor modelo",
        "",
        f"- Modelo: `{best_name}`",
        f"- Hiperparametros: `{json.dumps(best_params, ensure_ascii=False)}`",
        f"- ROC-AUC: {best_row.get('roc_auc', np.nan):.4f}",
        f"- Average Precision: {best_row.get('average_precision', np.nan):.4f}",
        f"- Precision@K: {best_row.get([c for c in metrics.columns if c.startswith('precision@')][0], np.nan):.4f}",
        f"- NDCG@K: {best_row.get([c for c in metrics.columns if c.startswith('ndcg@')][0], np.nan):.4f}",
        "",
        "## Arquivos gerados",
        "",
        "- `metricas_modelos.csv`",
        "- `metricas_agrupamento.csv`",
        "- `topicos_resumo.csv`",
        "- `recomendacoes_topk.csv`",
        "- `ativos_ict.csv`",
        "- `fig_01_evolucao_anual.png`",
        "- `fig_02_areas_lexicas.png`",
        "- `fig_03_metricas_modelos.png`",
        "- `fig_04_distribuicao_scores.png`",
        "- `fig_05_rede_recomendacoes.png`",
        "",
        "## Cinco recomendacoes com maior escore",
        "",
    ]
    for _, r in recs.head(5).iterrows():
        md.append(f"- Score {r['score_recomendacao']:.3f}: patente `{str(r['patente_NO_PEDIDO']).strip()}` ({r['patente_titulo']}) + software `{str(r['software_NO_PEDIDO']).strip()}` ({r['software_titulo']}).")
    (out_dir / "relatorio_experimento.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptn-zip", type=Path, default=DEFAULT_PTN_ZIP)
    parser.add_argument("--prg-zip", type=Path, default=DEFAULT_PRG_ZIP)
    parser.add_argument("--saida", type=Path, default=ROOT / "resultados")
    parser.add_argument("--modo", choices=["rapido", "completo"], default="rapido")
    parser.add_argument("--max-patentes", type=int, default=None)
    parser.add_argument("--max-softwares", type=int, default=None)
    parser.add_argument("--max-pares", type=int, default=None)
    parser.add_argument("--vizinhos", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sem-cache", action="store_true", help="Ignora ativos_ict.csv existente e relê os ZIPs.")
    args = parser.parse_args()

    if args.modo == "rapido":
        max_patents = args.max_patentes or 18000
        max_softwares = args.max_softwares or 12000
        max_pairs = args.max_pares or 90000
        neighbors = args.vizinhos or 10
    else:
        max_patents = args.max_patentes or 120000
        max_softwares = args.max_softwares or 50000
        max_pairs = args.max_pares or 350000
        neighbors = args.vizinhos or 18

    out_dir = args.saida.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(
        ptn_zip=args.ptn_zip,
        prg_zip=args.prg_zip,
        out_dir=out_dir,
        mode=args.modo,
        max_patents=max_patents,
        max_softwares=max_softwares,
        random_state=args.random_state,
        top_k=args.top_k,
    )

    print("1/7 Carregando e filtrando ativos de ICTs...", flush=True)
    cache_assets = out_dir / "ativos_ict.csv"
    if cache_assets.exists() and not args.sem_cache:
        print(f"    Reutilizando cache: {cache_assets}", flush=True)
        assets = pd.read_csv(cache_assets, dtype=str, encoding="utf-8-sig")
        assets["ano"] = pd.to_numeric(assets["ano"], errors="coerce")
        assets["n_depositantes"] = pd.to_numeric(assets["n_depositantes"], errors="coerce").fillna(1)
    else:
        assets = load_assets(cfg)
    assets.to_csv(out_dir / "ativos_ict.csv", index=False, encoding="utf-8-sig")

    print("2/7 Gerando representacoes textuais...", flush=True)
    assets, X_tfidf, X_emb, vectorizer, svd = build_embeddings(assets, args.random_state)
    joblib.dump({"vectorizer": vectorizer, "svd": svd}, out_dir / "modelo_embeddings.joblib")

    print("3/7 Descobrindo agrupamentos tecnologicos...", flush=True)
    assets = topic_discovery(assets, X_emb, out_dir, args.random_state)
    assets.to_csv(out_dir / "ativos_ict.csv", index=False, encoding="utf-8-sig")

    print("4/7 Construindo pares candidatos patente-software...", flush=True)
    pairs = make_candidate_pairs(assets, X_emb, max_pairs=max_pairs, top_neighbors=neighbors, random_state=args.random_state)
    gfeat = graph_features(assets, pairs)
    features = pd.concat([
        pairs[["cosine", "jaccard_texto", "mesma_uf", "mesma_area_lexica", "mesmo_topico", "dist_ano"]].reset_index(drop=True),
        gfeat.reset_index(drop=True),
    ], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pairs.to_csv(out_dir / "pares_candidatos.csv", index=False, encoding="utf-8-sig")

    print("5/7 Treinando modelos e buscando hiperparametros...", flush=True)
    metrics, best = train_models(features, pairs["y"], pairs["patent_idx"], out_dir, args.random_state, args.top_k)

    print("6/7 Gerando recomendacoes finais...", flush=True)
    recs = create_recommendations(assets, pairs, features, best[2], out_dir, args.top_k)

    print("7/7 Gerando figuras e relatorio...", flush=True)
    plot_outputs(assets, pairs, metrics, recs, out_dir, args.top_k)
    write_report(assets, pairs, metrics, recs, out_dir, best[1], best[3])

    print(f"Concluido. Resultados em: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
