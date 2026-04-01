import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from . import APP_STATE, RANDOM_STATE


def clean_binary_df(df, text_col, label_col):
    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].notna() & (df[text_col].str.len() > 0)]
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df[df[label_col].notna()]
    df[label_col] = df[label_col].astype(int)
    df = df[df[label_col].isin([0, 1])].reset_index(drop=True)
    return df


def balance_train_df(train_df, label_col, mode="upsample"):
    vc = train_df[label_col].value_counts()
    if len(vc) < 2:
        return train_df.copy()
    maj = vc.idxmax()
    minc = vc.idxmin()
    df_maj = train_df[train_df[label_col] == maj]
    df_min = train_df[train_df[label_col] == minc]
    if mode == "downsample":
        df_maj = df_maj.sample(n=len(df_min), random_state=RANDOM_STATE)
        out = pd.concat([df_maj, df_min], axis=0)
    else:
        df_min = df_min.sample(n=len(df_maj), replace=True, random_state=RANDOM_STATE)
        out = pd.concat([df_maj, df_min], axis=0)
    return out.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)
    out = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(labels, probs)
    except Exception:
        out["roc_auc"] = np.nan
    return out


def train_bertweet_once(
    csv_path: str,
    text_col: str,
    label_col: str,
    model_name: str = "vinai/bertweet-base",
    max_len: int = 128,
    epochs: int = 3,
    train_bs: int = 16,
    eval_bs: int = 32,
    lr: float = 2e-5,
    balance_train: bool = True,
    balance_mode: str = "upsample",
    output_dir: str = "./bertweet_runs",
):
    set_seed(RANDOM_STATE)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = clean_binary_df(df, text_col, label_col)
    train_val, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[label_col])
    train_df, val_df = train_test_split(train_val, test_size=0.2, random_state=RANDOM_STATE, stratify=train_val[label_col])
    if balance_train:
        train_df = balance_train_df(train_df, label_col, balance_mode)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

    def tok(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=max_len)

    ds_train = Dataset.from_pandas(train_df[[text_col, label_col]].rename(columns={label_col: "labels"}))
    ds_val = Dataset.from_pandas(val_df[[text_col, label_col]].rename(columns={label_col: "labels"}))
    ds_test = Dataset.from_pandas(test_df[[text_col, label_col]].rename(columns={label_col: "labels"}))
    ds_train = ds_train.map(tok, batched=True)
    ds_val = ds_val.map(tok, batched=True)
    ds_test = ds_test.map(tok, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        compute_metrics=compute_metrics_binary,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    val_pred = trainer.predict(ds_val)
    test_pred = trainer.predict(ds_test)

    def build_report(pred_obj, split_name):
        logits = pred_obj.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        preds = np.argmax(logits, axis=1)
        labels = pred_obj.label_ids
        metrics = {
            "split": split_name,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "roc_auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan,
        }
        cm = confusion_matrix(labels, preds)
        rep = classification_report(labels, preds, output_dict=True, zero_division=0)
        return metrics, cm, rep, probs, preds, labels

    val_metrics, _, _, _, _, _ = build_report(val_pred, "val")
    test_metrics, test_cm, _, test_probs, test_preds, _ = build_report(test_pred, "test")
    log_hist = pd.DataFrame(trainer.state.log_history)
    fig_loss = go.Figure()
    if "loss" in log_hist.columns:
        tmp = log_hist[log_hist["loss"].notna()].copy()
        if not tmp.empty:
            fig_loss.add_trace(go.Scatter(x=tmp["epoch"], y=tmp["loss"], mode="lines+markers", name="train_loss"))
    if "eval_loss" in log_hist.columns:
        tmp = log_hist[log_hist["eval_loss"].notna()].copy()
        if not tmp.empty:
            fig_loss.add_trace(go.Scatter(x=tmp["epoch"], y=tmp["eval_loss"], mode="lines+markers", name="val_loss"))
    fig_loss.update_layout(template="plotly_white", title="Криві навчання", xaxis_title="Епоха", yaxis_title="Втрата", height=420)
    fig_cm = px.imshow(
        test_cm,
        x=["Прогноз 0", "Прогноз 1"],
        y=["Істина 0", "Істина 1"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Матриця помилок (тест)",
    )
    fig_cm.update_layout(template="plotly_white", height=420)
    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    pred_test_df = test_df.copy().reset_index(drop=True)
    pred_test_df["prob_1"] = test_probs
    pred_test_df["pred"] = test_preds
    md = f"""
**Розмір train:** `{len(train_df)}`
**Розмір val:** `{len(val_df)}`
**Розмір test:** `{len(test_df)}`
**Модель:** `{model_name}`
"""
    APP_STATE["trained_model"] = trainer.model
    APP_STATE["trained_tokenizer"] = tokenizer
    return md, metrics_df, fig_loss, fig_cm, pred_test_df, log_hist
