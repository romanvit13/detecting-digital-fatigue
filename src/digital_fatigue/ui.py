import re

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from wordcloud import WordCloud

from . import APP_STATE
from .segmentation import run_segmentation
from .training import train_bertweet_once
from .utils import clean_keywords_string, normalize_for_topics


def build_wordcloud_figure(text: str, title: str):
    clean = normalize_for_topics(text)
    clean = re.sub(r"\b\d+\b", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    fig = go.Figure()
    if len(clean) < 3:
        fig.update_layout(template="plotly_white", title=title, height=320)
        return fig
    wc = WordCloud(width=900, height=420, background_color="white", collocations=False).generate(clean)
    fig = px.imshow(wc.to_array())
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(template="plotly_white", title=title, coloraxis_showscale=False, height=320)
    return fig


def build_segment_share_pies(by_segment: pd.DataFrame):
    n = len(by_segment)
    cols = 4 if n >= 4 else max(1, n)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "domain"}] * cols for _ in range(rows)],
        subplot_titles=[f"{r['segment_name']}" for _, r in by_segment.iterrows()],
    )
    for idx, (_, row) in enumerate(by_segment.iterrows(), start=1):
        rr = (idx - 1) // cols + 1
        cc = (idx - 1) % cols + 1
        fig.add_trace(
            go.Pie(
                labels=["Без виражених ознак", "Цифрова втома"],
                values=[100 - row["local_index"], row["local_index"]],
                textinfo="percent",
                showlegend=(idx == 1),
                sort=False,
            ),
            row=rr,
            col=cc,
        )
    fig.update_layout(
        template="plotly_white",
        height=min(max(420, 220 * rows), 900),
        margin=dict(l=30, r=30, t=70, b=30),
        title="Структура повідомлень у сегментах",
    )
    return fig


def build_author_profile_from_state(tau: float = 0.50, gamma: float = 50.0, alpha: float = 0.80):
    tau = 0.50 if tau is None else float(tau)
    gamma = 50.0 if gamma is None else float(gamma)
    alpha = 0.80 if alpha is None else float(alpha)
    seg_df = APP_STATE.get("segmentation_df")
    model = APP_STATE.get("trained_model")
    tokenizer = APP_STATE.get("trained_tokenizer")
    text_col = APP_STATE.get("text_col")
    user_col = APP_STATE.get("user_col")
    if seg_df is None or text_col is None or user_col is None:
        raise ValueError("Спочатку виконайте сегментацію повідомлень.")
    if model is None or tokenizer is None:
        raise ValueError("Спочатку навчіть класифікатор цифрової втоми.")
    work = seg_df.copy()
    work = work[work["cluster"] != -1].copy()
    if work.empty:
        raise ValueError("Після сегментації не залишилося ненульових сегментів.")
    texts = work[text_col].astype(str).fillna("").tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    probs_all = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        enc = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs_all.extend(probs.tolist())
    work["fatigue_prob"] = probs_all
    work["fatigue_bin"] = (work["fatigue_prob"] > tau).astype(int)
    work["segment_id"] = work["cluster"]
    by_segment = (
        work.groupby([user_col, "segment_id", "segment_name", "keywords"], dropna=False)
        .agg(
            n_messages=(text_col, "size"),
            mean_probability=("fatigue_prob", "mean"),
            local_index=("fatigue_bin", lambda x: float(np.mean(x) * 100.0)),
        )
        .reset_index()
        .rename(columns={user_col: "author_id", "keywords": "target_objects"})
        .sort_values("local_index", ascending=False)
        .reset_index(drop=True)
    )
    by_segment = by_segment.head(15).copy()
    by_segment["target_objects"] = by_segment["target_objects"].fillna("").map(clean_keywords_string)
    by_segment["critical"] = by_segment["local_index"] > gamma
    by_segment["segment_status"] = np.where(by_segment["critical"], "критичний", "некритичний")
    k = int(len(by_segment))
    m = int(by_segment["critical"].sum())
    coverage = float(m / k) if k > 0 else 0.0
    if coverage >= alpha:
        state_label = "Systemic Exhaustion"
        state_uk = "Системне цифрове виснаження"
    elif m > 0:
        state_label = "Situational Fatigue"
        state_uk = "Ситуативна цифрова втома"
    else:
        state_label = "Normal"
        state_uk = "Норма"
    author_id = by_segment["author_id"].iloc[0] if len(by_segment) else "невідомо"
    summary_md = f"""
## Інтегральний профіль цифрового виснаження

**Автор:** `{author_id}`
**Стан:** **{state_uk}** (`{state_label}`)
**CoverageΘ:** **{coverage:.3f}**
**Критичних сегментів:** **{m} із {k}**
**Пороги:** τ = `{tau:.2f}`, γ = `{gamma:.1f}`, α = `{alpha:.2f}`

Локальний індекс `eᵢ` = частка повідомлень сегмента, для яких оцінка моделі перевищує `τ`.
Інтегральний профіль автора формується через `CoverageΘ = count(eᵢ > γ) / k`.
"""
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=coverage * 100.0,
            number={"suffix": "%"},
            title={"text": "Охоплення критичних сегментів (CoverageΘ)"},
            gauge={
                "axis": {"range": [0, 100]},
                "threshold": {"line": {"color": "red", "width": 3}, "value": alpha * 100.0},
                "steps": [
                    {"range": [0, alpha * 100.0], "color": "#d8ead3"},
                    {"range": [alpha * 100.0, 100], "color": "#f4cccc"},
                ],
                "bar": {"color": "#4c6ef5"},
            },
        )
    )
    fig_gauge.update_layout(template="plotly_white", height=330, margin=dict(l=20, r=20, t=50, b=20))
    fig_segments = px.bar(
        by_segment.sort_values("local_index", ascending=True),
        x="local_index",
        y="segment_name",
        orientation="h",
        color="segment_status",
        hover_data=["n_messages", "mean_probability", "target_objects"],
        title="Локальні індекси цифрової втоми за сегментами",
    )
    fig_segments.add_vline(x=gamma, line_width=2, line_dash="dash", line_color="red")
    fig_segments.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title="Локальний індекс eᵢ, %",
        yaxis_title="Іменований сегмент",
    )
    fig_share = build_segment_share_pies(by_segment)
    top = by_segment.sort_values("local_index", ascending=False).head(3)
    wc_figs = []
    for _, row in top.iterrows():
        segid = row["segment_id"]
        txt = " ".join(work.loc[work["segment_id"] == segid, text_col].astype(str).tolist())
        wc_figs.append(build_wordcloud_figure(txt, f"Хмара слів: {row['segment_name']}"))
    while len(wc_figs) < 3:
        tmp = go.Figure()
        tmp.update_layout(template="plotly_white", height=320)
        wc_figs.append(tmp)
    out_table = by_segment[["segment_name", "target_objects", "n_messages", "mean_probability", "local_index", "segment_status"]].copy()
    out_table.columns = [
        "Іменований сегмент",
        "Цільові об'єкти / ключові терми",
        "К-сть повідомлень",
        "Сер. ймовірність",
        "Локальний індекс eᵢ, %",
        "Статус",
    ]
    APP_STATE["evaluated_segments"] = by_segment.copy()
    return summary_md, fig_gauge, fig_segments, out_table, fig_share, wc_figs[0], wc_figs[1], wc_figs[2]


def ui_train(csv_file, text_col, label_col, model_name, max_len, epochs, train_bs, eval_bs, lr, balance_train, balance_mode):
    csv_path = csv_file.name if hasattr(csv_file, "name") else csv_file
    return train_bertweet_once(
        csv_path=csv_path,
        text_col=text_col,
        label_col=label_col,
        model_name=model_name,
        max_len=int(max_len),
        epochs=int(epochs),
        train_bs=int(train_bs),
        eval_bs=int(eval_bs),
        lr=float(lr),
        balance_train=bool(balance_train),
        balance_mode=balance_mode,
    )


def ui_segmentation(
    csv_file,
    text_col,
    user_col,
    target_user,
    model_name,
    min_cluster_size,
    min_samples,
    top_n_keywords,
    lang,
    max_noise_share,
):
    csv_path = csv_file.name if hasattr(csv_file, "name") else csv_file
    return run_segmentation(
        csv_path=csv_path,
        text_col=text_col,
        user_col=user_col,
        target_user=target_user,
        model_name=model_name,
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        top_n_keywords=int(top_n_keywords),
        lang=lang,
        max_noise_share=None if max_noise_share is None or float(max_noise_share) <= 0 else float(max_noise_share),
    )


def ui_profile(tau, gamma, alpha):
    tau = 0.50 if tau is None or tau == "" else float(tau)
    gamma = 50.0 if gamma is None or gamma == "" else float(gamma)
    alpha = 0.80 if alpha is None or alpha == "" else float(alpha)
    return build_author_profile_from_state(tau=tau, gamma=gamma, alpha=alpha)


def create_app():
    with gr.Blocks(title="Інтерфейс експериментів для дисертації", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
    # Інтерфейс експериментів для дисертації
    **Логіка:** навчання класифікатора цифрової втоми → визначення комунікативних сегментів → інтегральний профіль цифрового виснаження автора.
    """
        )
        with gr.Tab("1. Навчання нейромережі цифрової втоми"):
            with gr.Row():
                csv_train = gr.File(label="CSV-файл для навчання", file_types=[".csv"])
                with gr.Column():
                    train_text_col = gr.Textbox(value="Tweet Text", label="Колонка з текстом")
                    train_label_col = gr.Textbox(value="burnout", label="Колонка з міткою 0/1")
                    train_model_name = gr.Textbox(value="vinai/bertweet-base", label="Модель")
                    max_len = gr.Slider(32, 256, value=128, step=8, label="Максимальна довжина")
                with gr.Column():
                    epochs = gr.Slider(1, 8, value=3, step=1, label="Кількість епох")
                    train_bs = gr.Slider(4, 32, value=16, step=4, label="Train batch size")
                    eval_bs = gr.Slider(4, 64, value=32, step=4, label="Eval batch size")
                    lr = gr.Number(value=2e-5, label="Learning rate")
                    balance_train = gr.Checkbox(value=True, label="Балансувати train")
                    balance_mode = gr.Dropdown(["upsample", "downsample"], value="upsample", label="Режим балансування")
            btn_train = gr.Button("Навчити модель", variant="primary")
            train_md = gr.Markdown()
            train_metrics = gr.Dataframe(label="Метрики")
            train_loss_fig = gr.Plot(label="Криві навчання")
            train_cm_fig = gr.Plot(label="Матриця помилок")
            train_pred_df = gr.Dataframe(label="Передбачення на тесті")
            train_log_df = gr.Dataframe(label="Лог навчання")
            btn_train.click(
                ui_train,
                inputs=[csv_train, train_text_col, train_label_col, train_model_name, max_len, epochs, train_bs, eval_bs, lr, balance_train, balance_mode],
                outputs=[train_md, train_metrics, train_loss_fig, train_cm_fig, train_pred_df, train_log_df],
            )
        with gr.Tab("2. Сегментація та інтегральний профіль автора"):
            gr.Markdown("### 2.1. Сегментація та виявлення цільових об’єктів")
            with gr.Row():
                csv_seg = gr.File(label="CSV-файл для сегментації", file_types=[".csv"])
                with gr.Column():
                    seg_text_col = gr.Textbox(value="post_text", label="Колонка з текстом")
                    seg_user_col = gr.Textbox(value="user_id", label="Колонка з автором")
                    seg_target_user = gr.Textbox(value="__AUTO__", label="ID автора або __AUTO__")
                    seg_model_name = gr.Textbox(value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", label="SentenceTransformer")
                    seg_lang = gr.Dropdown(["uk", "en"], value="uk", label="Мова стоп-слів")
                with gr.Column():
                    min_cluster_size = gr.Slider(5, 100, value=15, step=1, label="min_cluster_size")
                    min_samples = gr.Slider(1, 30, value=5, step=1, label="min_samples")
                    top_n_keywords = gr.Slider(5, 10, value=6, step=1, label="Кількість ключових термів")
                    max_noise_share = gr.Slider(0.0, 0.6, value=0.10, step=0.01, label="Макс. частка шуму (-1), 0 = вимкнено")
            btn_seg = gr.Button("Запустити сегментацію", variant="primary")
            seg_md = gr.Markdown()
            seg_umap_fig = gr.Plot(label="UMAP-схема сегментів")
            seg_dist_fig = gr.Plot(label="Теплова карта відстаней")
            seg_summary_df = gr.Dataframe(label="Іменовані сегменти (до 15)")
            seg_examples_df = gr.Dataframe(label="Приклади повідомлень")
            btn_seg.click(
                ui_segmentation,
                inputs=[
                    csv_seg,
                    seg_text_col,
                    seg_user_col,
                    seg_target_user,
                    seg_model_name,
                    min_cluster_size,
                    min_samples,
                    top_n_keywords,
                    seg_lang,
                    max_noise_share,
                ],
                outputs=[seg_md, seg_umap_fig, seg_dist_fig, seg_summary_df, seg_examples_df],
            )
            gr.Markdown("### 2.2. Інтегральний профіль цифрового виснаження")
            with gr.Row():
                tau = gr.Number(value=0.50, label="Поріг класифікації τ")
                gamma = gr.Number(value=50.0, label="Критичний поріг сегмента γ, %")
                alpha = gr.Number(value=0.80, label="Поріг системного поширення α")
            btn_profile = gr.Button("Побудувати інтегральний профіль", variant="primary")
            profile_md = gr.Markdown()
            profile_gauge = gr.Plot(label="CoverageΘ")
            profile_segments = gr.Plot(label="Локальні індекси сегментів")
            profile_table = gr.Dataframe(label="Таблиця сегментного профілю")
            profile_pies = gr.Plot(label="Структура повідомлень у сегментах")
            wc1 = gr.Plot(label="Хмара слів 1")
            wc2 = gr.Plot(label="Хмара слів 2")
            wc3 = gr.Plot(label="Хмара слів 3")
            btn_profile.click(
                ui_profile,
                inputs=[tau, gamma, alpha],
                outputs=[profile_md, profile_gauge, profile_segments, profile_table, profile_pies, wc1, wc2, wc3],
            )
    return app
