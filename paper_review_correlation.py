"""
PeerRead Subjectivity Scoring + Correlation Analysis

Steps:
1. Load PeerRead dataset (allenai/peer_read)
2. Score each review sentence using the best all-domain scorer
3. Aggregate subjectivity scores per review
4. Correlate review-level subjectivity with PeerRead aspect scores:
   - originality, clarity, soundness, correctness (where available)
5. Visualize and save results

Environment variables:
    export COMET_API_KEY="your_key"
    export COMET_PROJECT_NAME="peerread-subjectivity"
    export COMET_WORKSPACE="your_workspace"
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr, spearmanr
from comet_ml import Experiment
import re

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Path to the saved best model from all-domain training
BEST_MODEL_PATH  = "./all_domain_results_train/best_model/scorer_model.pt"
OUTPUT_DIR       = "./peerread_analysis"
BATCH_SIZE       = 64
MAX_LENGTH       = 256

# PeerRead aspect columns to correlate against
ASPECT_COLS = [
    "ORIGINALITY",
    "CLARITY",
    "SOUNDNESS_CORRECTNESS",
    "RECOMMENDATION",
    "SUBSTANCE",
    "MEANINGFUL_COMPARISON",
    "IMPACT",
]

COMET_API_KEY      = os.environ.get("COMET_API_KEY",      "muMGDSaYYC8xmekjTcGPSYePj")
COMET_PROJECT_NAME = os.environ.get("COMET_PROJECT_NAME", "peerread-subjectivity")
COMET_WORKSPACE    = os.environ.get("COMET_WORKSPACE",    "sanju-basak")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# MODEL (same architecture used for training)
# ─────────────────────────────────────────────
class TransformerSubjectivityScorer(nn.Module):
    def __init__(self, model_checkpoint: str, dropout: float = 0.1):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_checkpoint)
        hidden_size    = self.encoder.config.hidden_size
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits     = self.regressor(cls_output).squeeze(-1)
        return self.sigmoid(logits)


# ─────────────────────────────────────────────
# SENTENCE TOKENIZER
# ─────────────────────────────────────────────
def split_into_sentences(text: str):
    """Split review text into sentences using simple regex."""
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ─────────────────────────────────────────────
# SENTENCE DATASET FOR BATCH INFERENCE
# ─────────────────────────────────────────────
class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=MAX_LENGTH):
        self.sentences  = sentences
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ─────────────────────────────────────────────
# LOAD SCORER
# ─────────────────────────────────────────────
def load_scorer(model_path: str, device):
    print(f"Loading scorer from: {model_path}")
    checkpoint_data = torch.load(model_path, map_location=device)
    model_checkpoint = checkpoint_data["checkpoint"]
    print(f"Architecture : {checkpoint_data['model_name']}")
    print(f"Val RMSE     : {checkpoint_data['best_val_rmse']}")
    print(f"Train domains: {checkpoint_data['train_domains']}")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model     = TransformerSubjectivityScorer(model_checkpoint).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# SCORE SENTENCES IN BATCHES
# ─────────────────────────────────────────────
def score_sentences(sentences, model, tokenizer, device, batch_size=BATCH_SIZE):
    """Score a list of sentences and return 0-1 scores."""
    if not sentences:
        return []
    ds     = SentenceDataset(sentences, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size)
    scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds          = model(input_ids=input_ids, attention_mask=attention_mask)
            scores.extend(preds.cpu().numpy().tolist())
    return scores


# ─────────────────────────────────────────────
# AGGREGATE REVIEW SCORES
# ─────────────────────────────────────────────
def aggregate_scores(sentence_scores):
    """Compute review-level subjectivity aggregates from sentence scores."""
    if not sentence_scores:
        return None
    arr = np.array(sentence_scores)
    return {
        "subj_mean":         round(float(arr.mean()),  4),
        "subj_std":          round(float(arr.std()),   4),
        "subj_max":          round(float(arr.max()),   4),
        "subj_min":          round(float(arr.min()),   4),
        "subj_high_ratio":   round(float((arr >= 0.6).mean()), 4),  # % highly subjective
        "subj_low_ratio":    round(float((arr <= 0.3).mean()), 4),  # % highly objective
        "n_sentences":       len(sentence_scores),
    }


# ─────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────
def compute_correlations(review_records, aspect_cols):
    """
    Compute Pearson and Spearman correlation between
    review subjectivity aggregates and each aspect score.
    """
    subj_keys = ["subj_mean", "subj_std", "subj_high_ratio", "subj_low_ratio"]
    results   = {}

    for aspect in aspect_cols:
        aspect_vals = []
        subj_vals   = {k: [] for k in subj_keys}

        for rec in review_records:
            if rec.get(aspect) is None or rec.get("aggregates") is None:
                continue
            val = rec[aspect]
            if val is None or val == 0:
                continue
            aspect_vals.append(float(val))
            for k in subj_keys:
                subj_vals[k].append(rec["aggregates"][k])

        if len(aspect_vals) < 10:
            print(f"  Skipping {aspect} — only {len(aspect_vals)} valid samples")
            continue

        results[aspect] = {"n": len(aspect_vals)}
        for k in subj_keys:
            x = np.array(subj_vals[k])
            y = np.array(aspect_vals)
            try:
                p_r, p_p = pearsonr(x, y)
                s_r, s_p = spearmanr(x, y)
                results[aspect][k] = {
                    "pearson":       round(float(p_r), 4),
                    "pearson_pval":  round(float(p_p), 4),
                    "spearman":      round(float(s_r), 4),
                    "spearman_pval": round(float(s_p), 4),
                }
            except Exception:
                pass

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PeerRead Subjectivity Scoring + Correlation Analysis")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load scorer ───────────────────────────
    model, tokenizer = load_scorer(BEST_MODEL_PATH, device)

    # ── Load PeerRead ─────────────────────────
    print("\nLoading PeerRead dataset...")
    dataset = load_dataset("allenai/peer_read", "reviews", trust_remote_code=True)
    print(f"  Train  : {len(dataset['train'])}")
    print(f"  Val    : {len(dataset['validation'])}")
    print(f"  Test   : {len(dataset['test'])}")

    # ── Check available columns ───────────────
    sample = dataset["train"][0]
    print(f"\nAvailable columns: {list(sample.keys())}")

    # # Print aspect score ranges available
    # for col in ASPECT_COLS:
    #     vals = [ex.get(col) for ex in dataset["train"] if ex.get(col)]
    #     if vals:
    #         print(f"  {col}: {min(vals):.1f} – {max(vals):.1f} "
    #               f"(n={len(vals)})")
    #     else:
    #         print(f"  {col}: not found or all zero")

    # # ── Score all reviews ─────────────────────
    # print("\nScoring review sentences...")
    # review_records = []
    # total   = 0
    # skipped = 0

    # for split in ["train", "validation", "test"]:
    #     print(f"\n  Processing {split} split ({len(dataset[split])} reviews)...")
    #     for i, ex in enumerate(dataset[split]):
    #         review_text = ex.get("review_text", "") or \
    #                       ex.get("comments", "")   or \
    #                       ex.get("text", "")        or ""

    #         if not review_text.strip():
    #             skipped += 1
    #             continue

    #         sentences = split_into_sentences(review_text)
    #         if not sentences:
    #             skipped += 1
    #             continue

    #         sent_scores = score_sentences(sentences, model, tokenizer, device)
    #         aggregates  = aggregate_scores(sent_scores)

    #         record = {
    #             "split":            split,
    #             "paper_id":         ex.get("paper_id", ""),
    #             "review_idx":       i,
    #             "n_sentences":      len(sentences),
    #             "sentence_scores":  [round(s, 4) for s in sent_scores],
    #             "aggregates":       aggregates,
    #         }

    #         # Add aspect scores
    #         for col in ASPECT_COLS:
    #             val = ex.get(col)
    #             record[col] = float(val) if val else None

    #         review_records.append(record)
    #         total += 1

    #         if (i + 1) % 500 == 0:
    #             print(f"    {i+1} / {len(dataset[split])} processed...")

    # print(f"\n  Total scored  : {total}")
    # print(f"  Skipped       : {skipped}")

    # # ── Correlation analysis ──────────────────
    # print("\nComputing correlations...")
    # correlations = compute_correlations(review_records, ASPECT_COLS)

    # # Print correlation table
    # print("\n" + "="*80)
    # print("  CORRELATION: subj_mean vs each aspect score")
    # print("="*80)
    # print(f"{'Aspect':<28} {'Pearson':>10} {'P-val':>10} {'Spearman':>10} {'P-val':>10} {'N':>6}")
    # print("─" * 80)
    # for aspect, res in correlations.items():
    #     if "subj_mean" not in res:
    #         continue
    #     r  = res["subj_mean"]
    #     sig = "*" if r["pearson_pval"] < 0.05 else ""
    #     print(f"{aspect:<28} {r['pearson']:>10.4f}{sig:1} "
    #           f"{r['pearson_pval']:>9.4f} "
    #           f"{r['spearman']:>10.4f} "
    #           f"{r['spearman_pval']:>10.4f} "
    #           f"{res['n']:>6}")

    # print("\n  * p < 0.05")

    # # Print extended table for all subjectivity features
    # print("\n" + "="*80)
    # print("  PEARSON CORRELATIONS — all subjectivity features vs aspects")
    # print("="*80)
    # subj_keys = ["subj_mean", "subj_high_ratio", "subj_low_ratio", "subj_std"]
    # header = f"{'Aspect':<28}" + "".join(f"{k:>16}" for k in subj_keys)
    # print(header)
    # print("─" * 80)
    # for aspect, res in correlations.items():
    #     row = f"{aspect:<28}"
    #     for k in subj_keys:
    #         if k in res:
    #             row += f"{res[k]['pearson']:>16.4f}"
    #         else:
    #             row += f"{'N/A':>16}"
    #     print(row)

    # # ── Print overall statistics ──────────────
    # all_means = [r["aggregates"]["subj_mean"]
    #              for r in review_records if r["aggregates"]]
    # print(f"\n  Overall review subjectivity distribution:")
    # print(f"  Mean  : {np.mean(all_means):.4f}")
    # print(f"  Std   : {np.std(all_means):.4f}")
    # print(f"  Min   : {np.min(all_means):.4f}")
    # print(f"  Max   : {np.max(all_means):.4f}")
    # print(f"  % reviews with subj_mean > 0.6: "
    #       f"{100 * np.mean(np.array(all_means) > 0.6):.1f}%")
    # print(f"  % reviews with subj_mean < 0.3: "
    #       f"{100 * np.mean(np.array(all_means) < 0.3):.1f}%")

    # # ── Save results ──────────────────────────
    # # Full scored records
    # records_path = os.path.join(OUTPUT_DIR, "scored_reviews.json")
    # with open(records_path, "w") as f:
    #     json.dump(review_records, f, indent=2)
    # print(f"\nScored reviews saved → {records_path}")

    # # Correlation results
    # corr_path = os.path.join(OUTPUT_DIR, "correlations.json")
    # with open(corr_path, "w") as f:
    #     json.dump(correlations, f, indent=2)
    # print(f"Correlations saved  → {corr_path}")

    # # ── Log to CometML ────────────────────────
    # exp = Experiment(
    #     api_key=COMET_API_KEY,
    #     project_name=COMET_PROJECT_NAME,
    #     workspace=COMET_WORKSPACE,
    # )
    # exp.set_name("peerread-subjectivity-correlation")
    # exp.add_tags(["peerread", "correlation", "subjectivity"])
    # exp.log_parameter("n_reviews_scored", total)
    # exp.log_parameter("n_reviews_skipped", skipped)
    # exp.log_metric("mean_review_subjectivity", round(float(np.mean(all_means)), 4))

    # for aspect, res in correlations.items():
    #     if "subj_mean" in res:
    #         exp.log_metrics({
    #             f"{aspect}_pearson":  res["subj_mean"]["pearson"],
    #             f"{aspect}_spearman": res["subj_mean"]["spearman"],
    #         })
    # exp.end()
    # print("Logged to CometML ✓")
    # print("\nDone!")


if __name__ == "__main__":
    main()