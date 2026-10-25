"""
Transformer Model Comparison for Subjectivity Scoring
Compares: BERT-large, RoBERTa-large, DeBERTa-v3-large, DistilBERT
on SubjQA ans_subj_score regression task across all 6 domains

Metrics: MSE, RMSE, Pearson, Spearman
CometML tracking per model per domain

Environment variables:
    export COMET_API_KEY="your_key"
    export COMET_PROJECT_NAME="subjqa-transformer-comparison"
    export COMET_WORKSPACE="your_workspace"
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from comet_ml import Experiment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS = {
    "bert-large": {
        "checkpoint": "deepset/bert-large-uncased-whole-word-masking-squad2",
        "max_length": 256,
        "batch_size": 32,
    },
    "roberta-large": {
        "checkpoint": "roberta-large",
        "max_length": 256,
        "batch_size": 32,
    },
    "deberta-v3-large": {
        "checkpoint": "microsoft/deberta-v3-large",
        "max_length": 256,
        "batch_size": 16,   # DeBERTa-large needs more memory
    },
    "distilbert": {
        "checkpoint": "distilbert-base-uncased",
        "max_length": 256,
        "batch_size": 64,   # DistilBERT is small — use larger batch
    },
}

DOMAINS       = ["books", "electronics", "movies", "restaurants", "tripadvisor", "grocery"]
OUTPUT_DIR    = "./transformer_comparison_results_full"
EPOCHS        = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY  = 0.001

INTERACTIVE       = False   # ← flip to False for full run
INTERACTIVE_LIMIT = 64

COMET_API_KEY      = os.environ.get("COMET_API_KEY",      "muMGDSaYYC8xmekjTcGPSYePj")
COMET_PROJECT_NAME = os.environ.get("COMET_PROJECT_NAME", "subjqa-transformer-comparison-full")
COMET_WORKSPACE    = os.environ.get("COMET_WORKSPACE",    "sanju-basak")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class TransformerSubjectivityScorer(nn.Module):
    """
    Generic transformer regression head.
    Works for BERT, RoBERTa, DeBERTa, DistilBERT.
    [CLS] token → Dropout → Linear → Sigmoid → 0-1 score
    """
    def __init__(self, model_checkpoint: str, dropout: float = 0.1):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_checkpoint)
        hidden_size    = self.encoder.config.hidden_size
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token — works for all 4 model families
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits     = self.regressor(cls_output).squeeze(-1)
        score      = self.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(score, labels.float())

        return {"loss": loss, "logits": score}


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class SubjectivityDataset(Dataset):
    """
    Extracts all answer spans + ans_subj_score from SubjQA.
    Uses all answer spans per example to maximize training data.
    ans_subj_score is used directly (already 0-1).
    """
    def __init__(self, hf_dataset, tokenizer, max_length=256):
        self.samples = []
        for ex in hf_dataset:
            answers = ex["answers"]
            if not answers["text"] or len(answers["ans_subj_score"]) == 0:
                continue
            for sentence, score in zip(answers["text"], answers["ans_subj_score"]):
                sentence = sentence.strip()
                if not sentence:
                    continue
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
def compute_metrics(preds, labels):
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
def create_experiment(model_name: str, domain: str, phase: str) -> Experiment:
    exp = Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    exp.set_name(f"{model_name}-{domain}-{phase}")
    exp.add_tags([model_name, domain, phase, "subjectivity", "regression"])
    exp.log_parameters({
        "model":         model_name,
        "checkpoint":    MODELS[model_name]["checkpoint"],
        "domain":        domain,
        "phase":         phase,
        "max_length":    MODELS[model_name]["max_length"],
        "batch_size":    MODELS[model_name]["batch_size"],
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
def evaluate_model(model, dataloader, device, experiment=None, prefix=""):
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

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    if experiment:
        experiment.log_metrics({f"{prefix}{k}": v for k, v in metrics.items()})
    return metrics


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train_model(model_name, domain, train_dataset, val_dataset, checkpoint, batch_size):
    print(f"\n  Training {model_name.upper()} on {domain.upper()}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = TransformerSubjectivityScorer(checkpoint).to(device)
    save_dir = f"{OUTPUT_DIR}/{model_name}/{domain}"
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

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

    exp        = create_experiment(model_name, domain, "training")
    best_rmse  = float("inf")
    best_state = None
    step       = 0
    start_time = time.time()

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

        val_metrics = evaluate_model(model, val_loader, device,
                                     experiment=exp, prefix="val_")
        avg_loss    = np.mean(epoch_losses)
        print(f"    Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.4f} | "
              f"Pearson: {val_metrics['pearson']:.4f}")

        exp.log_metrics({"epoch_loss": avg_loss}, step=epoch+1)

        if val_metrics["rmse"] < best_rmse:
            best_rmse  = val_metrics["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    exp.end()

    # Save best model
    model.load_state_dict(best_state)
    train_time = round(time.time() - start_time, 2)
    torch.save({
        "model_state_dict": model.state_dict(),
        "checkpoint":       checkpoint,
        "model_name":       model_name,
        "domain":           domain,
        "best_val_rmse":    best_rmse,
        "train_time_sec":   train_time,
    }, f"{save_dir}/scorer_model.pt")
    print(f"    ✓ Saved | Best RMSE: {best_rmse:.4f} | Time: {train_time}s")

    return model, train_time


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  Transformer Comparison for Subjectivity Scoring")
    print(f"  Models : {', '.join(MODELS.keys())}")
    print(f"  Domains: {', '.join(DOMAINS)}")
    print(f"  Mode   : {'INTERACTIVE' if INTERACTIVE else 'FULL DATASET'}")
    print("="*70)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {model: {} for model in MODELS}

    for domain in DOMAINS:
        print(f"\n{'═'*70}")
        print(f"  DOMAIN: {domain.upper()}")
        print(f"{'═'*70}")

        # ── Load dataset once per domain ──────
        dataset    = load_dataset("megagonlabs/subjqa", domain)
        train_data = dataset["train"]
        val_data   = dataset["validation"]
        test_data  = dataset["test"]

        if INTERACTIVE:
            train_data = train_data.select(range(min(INTERACTIVE_LIMIT, len(train_data))))
            val_data   = val_data.select(range(min(INTERACTIVE_LIMIT, len(val_data))))
            test_data  = test_data.select(range(min(INTERACTIVE_LIMIT, len(test_data))))

        # ── Train and evaluate each model ─────
        for model_name, config in MODELS.items():
            print(f"\n{'─'*50}")
            print(f"  Model: {model_name.upper()}")
            print(f"{'─'*50}")

            checkpoint = config["checkpoint"]
            batch_size = config["batch_size"]
            max_length = config["max_length"]

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            # Build datasets
            train_dataset = SubjectivityDataset(train_data, tokenizer, max_length)
            val_dataset   = SubjectivityDataset(val_data,   tokenizer, max_length)
            test_dataset  = SubjectivityDataset(test_data,  tokenizer, max_length)
            test_loader   = DataLoader(test_dataset, batch_size=batch_size)

            print(f"  Samples → train={len(train_dataset)} | "
                  f"val={len(val_dataset)} | test={len(test_dataset)}")

            # ── BEFORE ────────────────────────
            pre_model  = TransformerSubjectivityScorer(checkpoint).to(device)
            before_exp = create_experiment(model_name, domain, "before")
            before_metrics = evaluate_model(
                pre_model, test_loader, device,
                experiment=before_exp, prefix="before_"
            )
            before_exp.end()
            print(f"  BEFORE → RMSE: {before_metrics['rmse']} | "
                  f"Pearson: {before_metrics['pearson']}")

            # Free memory before training
            del pre_model
            torch.cuda.empty_cache()

            # ── TRAIN ─────────────────────────
            trained_model, train_time = train_model(
                model_name, domain,
                train_dataset, val_dataset,
                checkpoint, batch_size,
            )

            # ── AFTER ─────────────────────────
            after_exp = create_experiment(model_name, domain, "after")
            after_metrics = evaluate_model(
                trained_model, test_loader, device,
                experiment=after_exp, prefix="after_"
            )
            after_exp.end()
            print(f"  AFTER  → RMSE: {after_metrics['rmse']} | "
                  f"Pearson: {after_metrics['pearson']}")

            all_results[model_name][domain] = {
                "before":     before_metrics,
                "after":      after_metrics,
                "delta": {
                    "rmse":     round(after_metrics["rmse"]     - before_metrics["rmse"],     4),
                    "pearson":  round(after_metrics["pearson"]  - before_metrics["pearson"],  4),
                    "spearman": round(after_metrics["spearman"] - before_metrics["spearman"], 4),
                },
                "train_time_sec": train_time,
            }

            # Free memory before next model
            del trained_model
            torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────
    print("\n" + "="*70)
    print("  FINAL COMPARISON SUMMARY (After Fine-tuning, Test Split)")
    print("="*70)
    print(f"{'Domain':<16} {'Model':<20} {'RMSE':>8} {'Pearson':>8} {'Spearman':>8} {'Time(s)':>8}")
    print("─" * 70)

    best_per_domain = {}
    for domain in DOMAINS:
        best_pearson = -999
        best_model   = None
        for model_name in MODELS:
            res        = all_results[model_name][domain]
            rmse       = res["after"]["rmse"]
            pearson    = res["after"]["pearson"]
            spearman   = res["after"]["spearman"]
            train_time = res["train_time_sec"]
            marker     = ""

            if pearson > best_pearson:
                best_pearson = pearson
                best_model   = model_name

            print(f"{domain:<16} {model_name:<20} {rmse:>8} {pearson:>8} {spearman:>8} {train_time:>8}")

        best_per_domain[domain] = best_model
        print()

    # ── Best model per domain ─────────────────
    print("\n" + "="*50)
    print("  BEST MODEL PER DOMAIN (by Pearson)")
    print("="*50)
    for domain, best in best_per_domain.items():
        pearson = all_results[best][domain]["after"]["pearson"]
        print(f"  {domain:<16} → {best:<20} (Pearson: {pearson})")

    # ── Save results ──────────────────────────
    results_path = f"{OUTPUT_DIR}/comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── Log summary to CometML ────────────────
    summary_exp = Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    summary_exp.set_name("summary-model-comparison")
    summary_exp.add_tag("summary")

    for model_name in MODELS:
        for domain in DOMAINS:
            res = all_results[model_name][domain]
            summary_exp.log_metrics({
                f"{model_name}_{domain}_rmse":     res["after"]["rmse"],
                f"{model_name}_{domain}_pearson":  res["after"]["pearson"],
                f"{model_name}_{domain}_spearman": res["after"]["spearman"],
                f"{model_name}_{domain}_time":     res["train_time_sec"],
            })
    summary_exp.end()
    print("Summary logged to CometML ✓")


if __name__ == "__main__":
    main()