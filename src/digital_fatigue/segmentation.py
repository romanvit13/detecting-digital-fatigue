from typing import List, Tuple

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from . import APP_STATE, RANDOM_STATE
from .utils import clean_keywords_string, normalize_for_topics, postprocess_terms, stopwords_for_lang, term_is_bad


def extract_keywords_ctfidf(
    texts: List[str],
    labels: List[int],
    top_n: int = 24,
    final_top_n: int = 8,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 3),
    lang: str = "uk",
) -> pd.DataFrame:
    tmp = pd.DataFrame({"text": texts, "cluster": labels})
    tmp = tmp[tmp["cluster"] != -1].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["cluster", "keywords"])
    agg = tmp.groupby("cluster")["text"].apply(lambda x: " ".join(map(normalize_for_topics, x))).reset_index()
    vect = CountVectorizer(
        stop_words=stopwords_for_lang(lang),
        min_df=min_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b[a-zA-ZА-Яа-яІіЇїЄєҐґ][a-zA-ZА-Яа-яІіЇїЄєҐґ'-]+\b",
    )
    X = vect.fit_transform(agg["text"])
    tf = X.toarray().astype(float)
    tf = tf / np.clip(tf.sum(axis=1, keepdims=True), 1e-9, None)
    df_term = (X > 0).sum(axis=0).A1
    idf = np.log((1 + X.shape[0]) / (1 + df_term)) + 1
    ctfidf = tf * idf[None, :]
    vocab = np.array(vect.get_feature_names_out())
    rows = []
    for i, cl in enumerate(agg["cluster"].tolist()):
        idx = np.argsort(-ctfidf[i])[:top_n]
        raw_terms = vocab[idx].tolist()
        clean_terms = postprocess_terms(raw_terms, max_terms=final_top_n)
        rows.append({"cluster": int(cl), "keywords": ", ".join(clean_terms)})
    return pd.DataFrame(rows)


def assign_segment_names(summary_df: pd.DataFrame, max_terms_in_name: int = 2) -> pd.DataFrame:
    out = summary_df.copy()
    out["keywords"] = out["keywords"].fillna("").map(clean_keywords_string)
    names = []
    for _, row in out.iterrows():
        kws = [p.strip() for p in str(row["keywords"]).split(",") if p.strip()]
        kws = [k for k in kws if not term_is_bad(k)]
        if len(kws) >= max_terms_in_name:
            base = " / ".join(kws[:max_terms_in_name])
        elif len(kws) == 1:
            base = kws[0]
        else:
            base = "невизначена тема"
        names.append(f"{base} [{int(row['cluster'])}]")
    out["segment_name"] = names
    return out


def cluster_examples(df_user, cluster_col="cluster", text_col="post_text", n_examples=3):
    rows = []
    for cl, sub in df_user[df_user[cluster_col] != -1].groupby(cluster_col):
        seg_name = sub["segment_name"].iloc[0] if "segment_name" in sub.columns else f"Сегмент {cl}"
        for t in sub[text_col].head(n_examples).tolist():
            rows.append({"cluster": int(cl), "segment_name": seg_name, "example": t})
    return pd.DataFrame(rows)


def pick_representative_texts(df: pd.DataFrame, cluster_id: int, text_col: str, prob_col: str = "cluster_prob", k: int = 3):
    sub = df[df["cluster"] == cluster_id].copy()
    if prob_col in sub.columns:
        sub = sub.sort_values(prob_col, ascending=False)
    texts = sub[text_col].astype(str).tolist()
    texts = [t for t in texts if len(t.strip()) >= 10]
    return texts[:k]


def separation_quick_stats(dist: np.ndarray, clusters: list):
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    nn_idx = d.argmin(axis=1)
    df_nn = pd.DataFrame(
        {
            "cluster": clusters,
            "nearest_cluster": [clusters[i] for i in nn_idx],
            "nearest_cosine_distance": nn,
        }
    ).sort_values("nearest_cosine_distance", ascending=False)
    stats = {
        "nearest_distance_median": float(np.median(nn)),
        "nearest_distance_min": float(np.min(nn)),
        "nearest_distance_max": float(np.max(nn)),
    }
    return df_nn, stats


def force_noise_to_max_share(embeddings: np.ndarray, labels_in: np.ndarray, max_noise_share: float = 0.10):
    labels_out = labels_in.copy()
    n_rows = len(labels_out)
    clusters = sorted([c for c in np.unique(labels_out) if c != -1])
    noise_idx = np.where(labels_out == -1)[0]
    if len(noise_idx) == 0 or len(clusters) == 0:
        return labels_out, float((labels_out == -1).mean()), None
    centroids = np.vstack([embeddings[labels_out == c].mean(axis=0) for c in clusters])
    dists = cosine_distances(embeddings[noise_idx], centroids)
    nearest_dist = dists.min(axis=1)
    nearest_cluster_idx = dists.argmin(axis=1)
    target_noise_count = int(np.floor(max_noise_share * n_rows))
    current_noise_count = len(noise_idx)
    if current_noise_count <= target_noise_count:
        return labels_out, current_noise_count / n_rows, None
    need_assign = current_noise_count - target_noise_count
    order = np.argsort(nearest_dist)
    assign_local = order[:need_assign]
    assign_global = noise_idx[assign_local]
    labels_out[assign_global] = np.array(clusters)[nearest_cluster_idx[assign_local]]
    tau = float(nearest_dist[assign_local].max())
    return labels_out, float((labels_out == -1).mean()), tau


def plot_clusters_clean(viz: pd.DataFrame, noise_label: int = -1):
    fig = go.Figure()
    for cl in sorted(viz["cluster"].unique()):
        sub = viz[viz["cluster"] == cl]
        name = "Шум" if cl == noise_label else f"Сегмент {cl}"
        fig.add_trace(go.Scattergl(x=sub["x"], y=sub["y"], mode="markers", name=name, text=sub["text_short"], marker=dict(size=7, opacity=0.78)))
    fig.update_layout(
        title="UMAP-представлення комунікативних сегментів",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        template="plotly_white",
        height=560,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def centroid_distance_heatmap(embeddings: np.ndarray, labels: np.ndarray):
    uniq = sorted([x for x in np.unique(labels) if x != -1])
    if len(uniq) < 2:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Недостатньо сегментів для теплової карти", height=420)
        return fig
    centroids = []
    for cl in uniq:
        centroids.append(embeddings[labels == cl].mean(axis=0))
    centroids = np.vstack(centroids)
    dist = cosine_distances(centroids)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    ordered_labels = [uniq[i] for i in order]
    ordered_dist = dist[np.ix_(order, order)]
    fig = px.imshow(
        ordered_dist,
        x=[f"Сегмент {c}" for c in ordered_labels],
        y=[f"Сегмент {c}" for c in ordered_labels],
        color_continuous_scale="Blues",
        title="Косинусні відстані між центроїдами сегментів",
    )
    fig.update_layout(template="plotly_white", height=480, margin=dict(l=30, r=30, t=60, b=30))
    return fig


def run_segmentation(
    csv_path: str,
    text_col: str,
    user_col: str,
    target_user: str,
    model_name: str,
    min_cluster_size: int,
    min_samples: int,
    top_n_keywords: int,
    lang: str = "uk",
    max_segments_display: int = 15,
    max_noise_share: float = None,
):
    df = pd.read_csv(csv_path)
    df = df[[user_col, text_col]].copy()
    df[text_col] = df[text_col].astype(str).fillna("").str.strip()
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    if target_user == "__AUTO__":
        user_stats = df.groupby(user_col).size().reset_index(name="n_posts").sort_values("n_posts", ascending=False)
        target_user_val = user_stats.iloc[0][user_col]
    else:
        target_user_val = pd.Series([target_user]).astype(df[user_col].dtype)[0]
    user_df = df[df[user_col] == target_user_val].copy().reset_index(drop=True)
    if len(user_df) < 10:
        raise ValueError("Для вибраного автора замало повідомлень для змістовної сегментації.")
    texts = user_df[text_col].tolist()
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    reducer10 = umap.UMAP(n_components=10, metric="cosine", random_state=RANDOM_STATE, n_neighbors=min(15, max(5, len(texts) // 10)))
    emb10 = reducer10.fit_transform(emb)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(emb10)
    probs = getattr(clusterer, "probabilities_", np.ones(len(labels)))
    noise_control_applied = max_noise_share is not None
    if noise_control_applied:
        labels, _, _ = force_noise_to_max_share(emb, labels, max_noise_share=float(max_noise_share))
    reducer2 = umap.UMAP(n_components=2, metric="cosine", random_state=RANDOM_STATE)
    emb2 = reducer2.fit_transform(emb)
    viz = pd.DataFrame({"x": emb2[:, 0], "y": emb2[:, 1], "cluster": labels, "prob": probs, "text_short": [t[:160] + ("..." if len(t) > 160 else "") for t in texts]})
    user_df["cluster"] = labels
    user_df["cluster_prob"] = probs
    keywords = extract_keywords_ctfidf(texts, labels, top_n=max(24, top_n_keywords * 3), final_top_n=top_n_keywords, lang=lang)
    summary = user_df.groupby("cluster").agg(n_messages=(text_col, "size"), mean_cluster_prob=("cluster_prob", "mean")).reset_index().sort_values(["cluster"])
    summary = summary.merge(keywords, on="cluster", how="left")
    summary = assign_segment_names(summary)
    summary_display = summary[summary["cluster"] != -1].sort_values(["n_messages", "mean_cluster_prob"], ascending=[False, False]).head(max_segments_display).copy()
    allowed_clusters = set(summary_display["cluster"].tolist())
    user_df = user_df.merge(summary[["cluster", "segment_name", "keywords"]], on="cluster", how="left")
    user_df_display = user_df[user_df["cluster"].isin(allowed_clusters)].copy()
    examples_rows = []
    for cl in sorted(user_df_display["cluster"].dropna().unique().tolist()):
        seg_name = user_df_display[user_df_display["cluster"] == cl]["segment_name"].iloc[0]
        for text in pick_representative_texts(user_df_display, int(cl), text_col=text_col, prob_col="cluster_prob", k=3):
            examples_rows.append({"cluster": int(cl), "segment_name": seg_name, "example": text})
    examples = pd.DataFrame(examples_rows)
    noise_share = float((labels == -1).mean())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    metrics_md = f"""
**Вибраний автор:** `{target_user_val}`
**Кількість повідомлень:** `{len(user_df)}`
**Кількість виявлених сегментів:** `{n_clusters}`
**Частка шуму:** `{noise_share:.3f}`
**Показано сегментів:** `{len(summary_display)}`
**Контроль шуму:** `{"увімкнено" if noise_control_applied else "вимкнено"}`
**Макс. частка шуму:** `{float(max_noise_share) if noise_control_applied else 0.0:.2f}`{"" if noise_control_applied else " (не застосовано)"}
"""
    fig_scatter = plot_clusters_clean(viz)
    fig_dist = centroid_distance_heatmap(emb, labels)
    APP_STATE["segmentation_df"] = user_df_display.copy()
    APP_STATE["segment_summary"] = summary_display.copy()
    APP_STATE["text_col"] = text_col
    APP_STATE["user_col"] = user_col
    APP_STATE["selected_user"] = target_user_val
    return metrics_md, fig_scatter, fig_dist, summary_display, examples
