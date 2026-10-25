"""
Subjectivity Scorer — Part 1
- Fine-tunes BERT on SubjQA ans_subj_score (0-1) as a regression task
- Predicts a 0-1 subjectivity score for any input sentence
- Domain-specific models trained per domain
- CometML tracking
- Evaluates on test split with MSE, RMSE, Pearson, Spearman

Environment variables:
    export COMET_API_KEY="your_key"
    export COMET_PROJECT_NAME="subjqa-subjectivity"
    export COMET_WORKSPACE="your_workspace"
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from comet_ml import Experiment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SCORER_MODEL  = "deepset/bert-large-uncased-whole-word-masking-squad2"
DOMAINS       = ["books", "electronics", "movies", "restaurants", "tripadvisor", "grocery"]
OUTPUT_DIR    = "./subjectivity_results_bert_regression"
MAX_LENGTH    = 256
BATCH_SIZE    = 32
EPOCHS        = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY  = 0.001

INTERACTIVE       = False   # ← flip to False for full run
INTERACTIVE_LIMIT = 64

COMET_API_KEY      = os.environ.get("COMET_API_KEY",      "muMGDSaYYC8xmekjTcGPSYePj")
COMET_PROJECT_NAME = os.environ.get("COMET_PROJECT_NAME", "subjqa-bert-regression")
COMET_WORKSPACE    = os.environ.get("COMET_WORKSPACE",    "sanju-basak")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# BERT REGRESSION MODEL
# ─────────────────────────────────────────────
class BERTSubjectivityScorer(nn.Module):
    """
    BERT with a regression head on top.
    Takes a sentence, outputs a single 0-1 subjectivity score.

    Architecture:
        BERT [CLS] token → Dropout → Linear(hidden_size → 1) → Sigmoid
    """
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(model_name)
        hidden_size    = self.bert.config.hidden_size  # 768 base / 1024 large
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits     = self.regressor(cls_output).squeeze(-1)
        score      = self.sigmoid(logits)                # constrain to 0-1

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(score, labels.float())

        return {"loss": loss, "logits": score}


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class SubjectivityDataset(Dataset):
    """
    Extracts answer spans + subjectivity labels from SubjQA.

    Uses ans_subj_score directly as the regression target because:
    - Already normalized to 0-1 range (min=0.0, max=1.0, mean=0.48)
    - Independently annotated from answer_subj_level
    - answer_subj_level (1-5) has a non-linear, non-monotonic mapping
      to subjectivity (verified empirically):
        level 1 -> 0.42 | level 2 -> 1.00 | level 3 -> 0.47
        level 4 -> 0.62 | level 5 -> 0.00
      So simple normalization (level-1)/4 would be incorrect.

    All answer spans are used (not just first) to maximize training data:
        books        : 445 -> 629  (+41%)
        electronics  : 511 -> 760  (+49%)
        movies       : 513 -> 754  (+47%)
        restaurants  : 598 -> 871  (+46%)
        tripadvisor  : 686 -> 1095 (+60%)
        grocery      : 523 -> 786  (+50%)
    """
    def __init__(self, hf_dataset, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        for ex in hf_dataset:
            answers = ex["answers"]

            # Skip examples with no answers
            if not answers["text"] or len(answers["ans_subj_score"]) == 0:
                continue

            # Loop over ALL answer spans — each becomes a training example
            for sentence, score in zip(answers["text"], answers["ans_subj_score"]):
                sentence = sentence.strip()
                if not sentence:
                    continue
                # 0.0 = fully objective, 1.0 = fully subjective
                self.samples.append((sentence, float(score)))

        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, score = self.samples[idx]
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(score, dtype=torch.float),
        }


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_regression_metrics(preds, labels):
    """Compute MSE, RMSE, Pearson r, Spearman r."""
    mse      = mean_squared_error(labels, preds)
    rmse     = np.sqrt(mse)
    pearson  = pearsonr(preds, labels)[0]
    spearman = spearmanr(preds, labels)[0]
    return {
        "mse":      round(float(mse),      4),
        "rmse":     round(float(rmse),     4),
        "pearson":  round(float(pearson),  4),
        "spearman": round(float(spearman), 4),
    }


# ─────────────────────────────────────────────
# COMET HELPER
# ─────────────────────────────────────────────
def create_experiment(domain: str, phase: str) -> Experiment:
    exp = Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    exp.set_name(f"{domain}-{phase}")
    exp.add_tags([domain, phase, "subjectivity", "regression", "bert-large"])
    exp.log_parameters({
        "model":         SCORER_MODEL,
        "domain":        domain,
        "phase":         phase,
        "max_length":    MAX_LENGTH,
        "batch_size":    BATCH_SIZE,
        "epochs":        EPOCHS,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio":  WARMUP_RATIO,
        "weight_decay":  WEIGHT_DECAY,
        "interactive":   INTERACTIVE,
    })
    return exp


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate_scorer(model, dataloader, device, experiment=None, prefix=""):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].numpy()
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = outputs["logits"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    metrics = compute_regression_metrics(
        np.array(all_preds), np.array(all_labels)
    )
    if experiment:
        experiment.log_metrics({f"{prefix}{k}": v for k, v in metrics.items()})
    return metrics


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train_scorer(domain, train_dataset, val_dataset, tokenizer):
    print(f"\n{'='*60}")
    print(f"  Training Subjectivity Scorer: {domain.upper()}")
    print(f"  Train : {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"{'='*60}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = BERTSubjectivityScorer(SCORER_MODEL).to(device)
    save_dir = f"{OUTPUT_DIR}/{domain}"
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    exp       = create_experiment(domain, "training")
    best_rmse = float("inf")
    best_state= None
    step      = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            step += 1

            if step % 10 == 0:
                exp.log_metrics({
                    "train_loss":    loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }, step=step)

        # ── Validate ──
        val_metrics = evaluate_scorer(model, val_loader, device,
                                      experiment=exp, prefix="val_")
        avg_loss    = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.4f} | "
              f"Pearson: {val_metrics['pearson']:.4f} | "
              f"Spearman: {val_metrics['spearman']:.4f}")

        exp.log_metrics({"epoch_loss": avg_loss}, step=epoch+1)

        # ── Save best model ──
        if val_metrics["rmse"] < best_rmse:
            best_rmse  = val_metrics["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best RMSE: {best_rmse:.4f} — model saved")

    exp.end()

    # ── Restore and save best model ──
    model.load_state_dict(best_state)
    model_path = f"{save_dir}/scorer_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name":       SCORER_MODEL,
        "domain":           domain,
        "best_val_rmse":    best_rmse,
    }, model_path)
    print(f"\n  Model saved → {model_path}")
    return model


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def get_subjectivity_score(sentence: str, model, tokenizer, device) -> float:
    """Given a sentence, return a 0-1 subjectivity score."""
    model.eval()
    encoding = tokenizer(
        sentence,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )
    return round(float(outputs["logits"].cpu().item()), 4)


def load_scorer(domain: str, device):
    """Load a saved scorer model for a given domain."""
    tokenizer  = AutoTokenizer.from_pretrained(SCORER_MODEL)
    checkpoint = torch.load(
        f"{OUTPUT_DIR}/{domain}/scorer_model.pt",
        map_location=device,
    )
    model = BERTSubjectivityScorer(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  SubjQA Subjectivity Scorer Training")
    print(f"  Model  : {SCORER_MODEL}")
    print(f"  Mode   : {'INTERACTIVE' if INTERACTIVE else 'FULL DATASET'}")
    print(f"  CometML: {COMET_PROJECT_NAME}")
    print("="*60)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(SCORER_MODEL)
    all_results = {}

    for domain in DOMAINS:
        print(f"\n{'─'*50}")
        print(f"  Domain: {domain.upper()}")
        print(f"{'─'*50}")

        dataset = load_dataset("megagonlabs/subjqa", domain)

        train_data = dataset["train"]
        val_data   = dataset["validation"]
        test_data  = dataset["test"]

        if INTERACTIVE:
            train_data = train_data.select(range(min(INTERACTIVE_LIMIT, len(train_data))))
            val_data   = val_data.select(range(min(INTERACTIVE_LIMIT, len(val_data))))
            test_data  = test_data.select(range(min(INTERACTIVE_LIMIT, len(test_data))))

        train_dataset = SubjectivityDataset(train_data, tokenizer)
        val_dataset   = SubjectivityDataset(val_data,   tokenizer)
        test_dataset  = SubjectivityDataset(test_data,  tokenizer)

        print(f"  Samples → train={len(train_dataset)} | "
              f"val={len(val_dataset)} | test={len(test_dataset)}")

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # ── BEFORE fine-tuning ────────────────
        print(f"\n[{domain}] Evaluating BEFORE fine-tuning...")
        pre_model  = BERTSubjectivityScorer(SCORER_MODEL).to(device)
        before_exp = create_experiment(domain, "before")
        before_metrics = evaluate_scorer(
            pre_model, test_loader, device,
            experiment=before_exp, prefix="before_"
        )
        before_exp.end()
        print(f"  BEFORE → RMSE: {before_metrics['rmse']} | "
              f"Pearson: {before_metrics['pearson']} | "
              f"Spearman: {before_metrics['spearman']}")

        # ── Train ─────────────────────────────
        scorer = train_scorer(domain, train_dataset, val_dataset, tokenizer)

        # ── AFTER fine-tuning ─────────────────
        print(f"\n[{domain}] Evaluating AFTER fine-tuning...")
        after_exp = create_experiment(domain, "after")
        after_metrics = evaluate_scorer(
            scorer, test_loader, device,
            experiment=after_exp, prefix="after_"
        )
        after_exp.end()
        print(f"  AFTER  → RMSE: {after_metrics['rmse']} | "
              f"Pearson: {after_metrics['pearson']} | "
              f"Spearman: {after_metrics['spearman']}")

        all_results[domain] = {
            "before": before_metrics,
            "after":  after_metrics,
            "delta": {
                "rmse":     round(after_metrics["rmse"]     - before_metrics["rmse"],     4),
                "pearson":  round(after_metrics["pearson"]  - before_metrics["pearson"],  4),
                "spearman": round(after_metrics["spearman"] - before_metrics["spearman"], 4),
            },
        }

    # ── Summary table ─────────────────────────
    print("\n" + "="*76)
    print("  FINAL RESULTS SUMMARY")
    print("="*76)
    print(f"{'Domain':<16} {'B-RMSE':>8} {'A-RMSE':>8} "
          f"{'B-Pear':>8} {'A-Pear':>8} "
          f"{'B-Spear':>8} {'A-Spear':>8}")
    print("─" * 76)
    for domain, res in all_results.items():
        print(
            f"{domain:<16}"
            f"{res['before']['rmse']:>8}"
            f"{res['after']['rmse']:>8}"
            f"{res['before']['pearson']:>8}"
            f"{res['after']['pearson']:>8}"
            f"{res['before']['spearman']:>8}"
            f"{res['after']['spearman']:>8}"
        )

    # ── Save JSON ─────────────────────────────
    results_path = f"{OUTPUT_DIR}/scorer_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── Quick inference demo ──────────────────
    print("\n" + "="*60)
    print("  INFERENCE DEMO")
    print("="*60)
    test_sentences = [
        ("books",       "This is a wonderfully written book."),
        ("electronics", "The battery lasts approximately 8 hours."),
        ("movies",      "This film completely changed my life."),
        ("restaurants", "The pasta was overcooked and the sauce was bland."),
        ("tripadvisor", "The room was exactly as described on the website."),
        ("grocery",     "This is the best coffee I have ever tasted."),
    ]
    for domain, sentence in test_sentences:
        scorer, tok = load_scorer(domain, device)
        score       = get_subjectivity_score(sentence, scorer, tok, device)
        label       = "subjective" if score >= 0.5 else "objective"
        print(f"  [{domain}] {sentence}")
        print(f"  Score: {score:.4f} ({label})\n")

    # ── Log summary to CometML ────────────────
    summary_exp = Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    summary_exp.set_name("summary-all-domains")
    summary_exp.add_tag("summary")
    for domain, res in all_results.items():
        summary_exp.log_metrics({
            f"{domain}_before_rmse":    res["before"]["rmse"],
            f"{domain}_after_rmse":     res["after"]["rmse"],
            f"{domain}_before_pearson": res["before"]["pearson"],
            f"{domain}_after_pearson":  res["after"]["pearson"],
            f"{domain}_delta_rmse":     res["delta"]["rmse"],
            f"{domain}_delta_pearson":  res["delta"]["pearson"],
        })
    summary_exp.end()
    print("Summary logged to CometML ✓")


if __name__ == "__main__":
    main()