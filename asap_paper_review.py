"""
ICLR 2017 Review Subjectivity Scoring + Correlation Analysis (Multi-file version)

Dataset: ASAP-Review (Kaggle) — ICLR_2017/ICLR_2017_review/*.json
"""

import os
import json
import re
import glob
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr, spearmanr
from comet_ml import Experiment

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Updated to point to the directory containing multiple JSON files
DATA_DIR         = "./asap-review/ICLR_2017/ICLR_2017_review" 
BEST_MODEL_PATH  = "./all_domain_results_train/best_model/scorer_model.pt"
OUTPUT_DIR       = "./iclr_analysis"
BATCH_SIZE       = 64
MAX_LENGTH       = 256

COMET_API_KEY      = os.environ.get("COMET_API_KEY",      "muMGDSaYYC8xmekjTcGPSYePj")
COMET_PROJECT_NAME = os.environ.get("COMET_PROJECT_NAME", "iclr-subjectivity")
COMET_WORKSPACE    = os.environ.get("COMET_WORKSPACE",    "sanju-basak")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# PARSE RATING AND CONFIDENCE
# ─────────────────────────────────────────────
def parse_rating(rating_str: str):
    if not rating_str:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[:\-]", rating_str.strip())
    if match:
        return float(match.group(1))
    match = re.match(r"^\s*(\d+(?:\.\d+)?)", rating_str.strip())
    return float(match.group(1)) if match else None


def parse_confidence(conf_str: str):
    if not conf_str:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[:\-]", conf_str.strip())
    if match:
        return float(match.group(1))
    match = re.match(r"^\s*(\d+(?:\.\d+)?)", conf_str.strip())
    return float(match.group(1)) if match else None


# ─────────────────────────────────────────────
# MODEL
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
        return self.sigmoid(self.regressor(cls_output).squeeze(-1))


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
# HELPERS
# ─────────────────────────────────────────────
def split_into_sentences(text: str):
    if not text or not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def load_scorer(model_path: str, device):
    print(f"Loading scorer from : {model_path}")
    ckpt       = torch.load(model_path, map_location=device)
    checkpoint = ckpt["checkpoint"]
    print(f"  Architecture : {ckpt['model_name']}")
    print(f"  Val RMSE     : {ckpt['best_val_rmse']}")
    tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
    model      = TransformerSubjectivityScorer(checkpoint).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


def score_sentences(sentences, model, tokenizer, device):
    if not sentences:
        return []
    ds     = SentenceDataset(sentences, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    scores = []
    with torch.no_grad():
        for batch in loader:
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )
            scores.extend(preds.cpu().numpy().tolist())
    return scores


def aggregate_scores(scores):
    if not scores:
        return None
    arr = np.array(scores)
    return {
        "subj_mean":       round(float(arr.mean()),                    4),
        "subj_std":        round(float(arr.std()),                     4),
        "subj_max":        round(float(arr.max()),                     4),
        "subj_min":        round(float(arr.min()),                     4),
        "subj_high_ratio": round(float((arr >= 0.6).mean()),           4),
        "subj_low_ratio":  round(float((arr <= 0.3).mean()),           4),
        "n_sentences":     len(scores),
    }


def correlate(x_vals, y_vals, label_x, label_y):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    p_r, p_p = pearsonr(x, y)
    s_r, s_p = spearmanr(x, y)
    return {
        "x":             label_x,
        "y":             label_y,
        "n":             len(x),
        "pearson":       round(float(p_r), 4),
        "pearson_pval":  round(float(p_p), 6),
        "spearman":      round(float(s_r), 4),
        "spearman_pval": round(float(s_p), 6),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  ICLR 2017 Subjectivity Scoring + Correlation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load scorer ───────────────────────────
    model, tokenizer = load_scorer(BEST_MODEL_PATH, device)

    # ── Load All ICLR 2017 Data Files ─────────
    print(f"\nSearching for JSON files in: {DATA_DIR}")
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    
    if not json_files:
        print(f"Error: No JSON files found in {DATA_DIR}. Please check the path.")
        return

    print(f"  Found {len(json_files)} files. Loading data...")
    papers = []
    for file_path in json_files:
        with open(file_path, "r") as f:
            try:
                raw = json.load(f)
                if isinstance(raw, dict):
                    papers.append(raw)
                elif isinstance(raw, list):
                    papers.extend(raw)
                else:
                    print(f"  Warning: Skipping {file_path} because JSON root is {type(raw).__name__}")
            except Exception as e:
                print(f"  Warning: Skipping {file_path} due to error: {e}")

    print(f"  Total papers loaded : {len(papers)}")
    total_reviews_count = sum(len(p.get("reviews", [])) for p in papers)
    print(f"  Total reviews found : {total_reviews_count}")

    # ── Quick parse verification ──────────────
    if papers and papers[0].get("reviews"):
        print("\nVerifying rating/confidence parsing on sample reviews...")
        for rev in papers[0].get("reviews", [])[:2]:
            r = parse_rating(rev.get("rating", ""))
            c = parse_confidence(rev.get("confidence", ""))
            print(f"  rating str : {rev.get('rating', '')[:50]}...")
            print(f"  parsed     : {r}")
            print(f"  conf str   : {rev.get('confidence', '')[:50]}...")
            print(f"  parsed     : {c}\n")

    # ── Score all reviews ─────────────────────
    print("Scoring reviews...")
    review_records = []
    skipped = 0

    for pi, paper in enumerate(papers):
        paper_id = paper.get("id", f"paper_{pi}")
        reviews  = paper.get("reviews", [])

        for ri, rev in enumerate(reviews):
            review_text = rev.get("review", "").strip()
            rating      = parse_rating(rev.get("rating", ""))
            confidence  = parse_confidence(rev.get("confidence", ""))

            if not review_text:
                skipped += 1
                continue

            sentences   = split_into_sentences(review_text)
            if not sentences:
                skipped += 1
                continue

            sent_scores = score_sentences(sentences, model, tokenizer, device)
            aggregates  = aggregate_scores(sent_scores)

            review_records.append({
                "paper_id":        paper_id,
                "review_idx":      ri,
                "rating":          rating,
                "confidence":      confidence,
                "n_sentences":     len(sentences),
                "sentence_scores": [round(s, 4) for s in sent_scores],
                "aggregates":      aggregates,
            })

        if (pi + 1) % 50 == 0:
            print(f"  {pi+1} / {len(papers)} papers processed...")

    print(f"\n  Reviews scored  : {len(review_records)}")
    print(f"  Reviews skipped : {skipped}")

    if not review_records:
        print("No reviews were successfully scored. Exiting.")
        return

    # ── Data extraction for analysis ──────────
    valid_ratings_data = [r for r in review_records if r["rating"] is not None]
    valid_conf_data    = [r for r in review_records if r["confidence"] is not None]

    ratings     = [r["rating"] for r in valid_ratings_data]
    confidences = [r["confidence"] for r in valid_conf_data]

    # ── Subjectivity distribution ─────────────
    all_means = [r["aggregates"]["subj_mean"] for r in review_records]
    print(f"\n  Subjectivity distribution:")
    print(f"  Mean  : {np.mean(all_means):.4f} | Std: {np.std(all_means):.4f}")
    
    # ── Correlation analysis ──────────────────
    subj_keys = ["subj_mean", "subj_high_ratio", "subj_low_ratio", "subj_std"]
    targets   = [("rating", ratings), ("confidence", confidences)]
    
    all_correlations = {}

    print("\n" + "="*72)
    print("  CORRELATION RESULTS")
    print("="*72)

    for target_name in ["rating", "confidence"]:
        print(f"\n  Target: {target_name.upper()}")
        all_correlations[target_name] = {}

        for sk in subj_keys:
            # Re-align for each specific subjectivity key and target
            x_vals = [r["aggregates"][sk] for r in review_records if r[target_name] is not None]
            y_vals = [r[target_name] for r in review_records if r[target_name] is not None]
            
            if len(x_vals) < 5: continue

            res = correlate(x_vals, y_vals, sk, target_name)
            all_correlations[target_name][sk] = res
            sig = "*" if res["pearson_pval"] < 0.05 else " "
            print(f"  {sk:<22} Pearson: {res['pearson']:>7.4f}{sig} (p={res['pearson_pval']:.4f})")

    # ── Save ─────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "scored_reviews.json"), "w") as f:
        json.dump(review_records, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        summary = {
            "n_papers": len(papers),
            "n_reviews": len(review_records),
            "correlations": all_correlations
        }
        json.dump(summary, f, indent=2)

    # ── CometML ──────────────────────────────
    try:
        exp = Experiment(api_key=COMET_API_KEY, project_name=COMET_PROJECT_NAME, workspace=COMET_WORKSPACE)
        exp.set_name("iclr2017-multi-file-analysis")
        exp.log_parameters({"n_files": len(json_files), "n_papers": len(papers)})
        exp.log_metric("mean_subjectivity", np.mean(all_means))
        exp.end()
        print("\nLogged to CometML ✓")
    except Exception as e:
        print(f"\nCometML logging failed (check your API key): {e}")

    print("Done!")

if __name__ == "__main__":
    main()