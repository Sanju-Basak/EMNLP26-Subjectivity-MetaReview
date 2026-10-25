"""
Comprehensive Cross-Domain + In-Domain Subjectivity Scoring

For each held-out domain X and each model:
    Train  : all domains EXCEPT X (5 domains combined)
    Validate: all domains EXCEPT X (validation splits combined)
    Test on:
        1. Held-out domain X test split       (cross-domain)
        2. Each training domain separately    (in-domain per domain)
        3. All training domains combined      (in-domain combined)
        4. All 6 domains combined             (overall)

This reveals:
    - How well the model generalizes to an unseen domain
    - How well it retains performance on seen domains
    - Which domain is hardest to generalize to
    - Which model is most domain-agnostic

Models : BERT-large, RoBERTa-large, DeBERTa-v3-base, DistilBERT
Metrics: RMSE, Pearson, Spearman

Environment variables:
    export COMET_API_KEY="your_key"
    export COMET_PROJECT_NAME="subjqa-cross-domain"
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
    "deberta-v3-base": {
        "checkpoint": "microsoft/deberta-v3-base",
        "max_length": 256,
        "batch_size": 16,
    },
    "distilbert": {
        "checkpoint": "distilbert-base-uncased",
        "max_length": 256,
        "batch_size": 64,
    },
}

DOMAINS       = ["books", "electronics", "movies", "restaurants", "tripadvisor", "grocery"]
OUTPUT_DIR    = "./cross_domain_results_full"
EPOCHS        = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY  = 0.001

INTERACTIVE       = False   # ← flip to False for full run
INTERACTIVE_LIMIT = 64

COMET_API_KEY      = os.environ.get("COMET_API_KEY",      "muMGDSaYYC8xmekjTcGPSYePj")
COMET_PROJECT_NAME = os.environ.get("COMET_PROJECT_NAME", "subjqa-cross-domain-test")
COMET_WORKSPACE    = os.environ.get("COMET_WORKSPACE",    "sanju-basak")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class TransformerSubjectivityScorer(nn.Module):
    """
    Generic transformer regression head.
    [CLS] → Dropout → Linear → Sigmoid → 0-1 score
    Works for BERT, RoBERTa, DeBERTa, DistilBERT.
    """
    def __init__(self, model_checkpoint: str, dropout: float = 0.1):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_checkpoint)
        hidden_size    = self.encoder.config.hidden_size
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs    = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
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
    ans_subj_score used directly (already 0-1).
    0.0 = fully objective, 1.0 = fully subjective
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
def create_experiment(model_name: str, held_out: str, phase: str) -> Experiment:
    exp = Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    exp.set_name(f"{model_name}-heldout={held_out}-{phase}")
    exp.add_tags([model_name, held_out, phase, "cross-domain", "subjqa"])
    exp.log_parameters({
        "model":         model_name,
        "checkpoint":    MODELS[model_name]["checkpoint"],
        "held_out":      held_out,
        "train_domains": [d for d in DOMAINS if d != held_out],
        "phase":         phase,
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
def train_model(model_name, held_out, train_dataset,
                val_dataset, checkpoint, batch_size):
    train_domains = [d for d in DOMAINS if d != held_out]
    print(f"\n  [{model_name}] Train on: {train_domains}")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = TransformerSubjectivityScorer(checkpoint).to(device)
    save_dir = f"{OUTPUT_DIR}/{model_name}/heldout_{held_out}"
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

    exp        = create_experiment(model_name, held_out, "training")
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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
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

        val_metrics = evaluate_model(
            model, val_loader, device,
            experiment=exp, prefix="val_"
        )
        avg_loss = np.mean(epoch_losses)
        print(f"    Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.4f} | "
              f"Pearson: {val_metrics['pearson']:.4f}")

        exp.log_metrics({"epoch_loss": avg_loss}, step=epoch+1)

        if val_metrics["rmse"] < best_rmse:
            best_rmse  = val_metrics["rmse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    exp.end()

    model.load_state_dict(best_state)
    train_time = round(time.time() - start_time, 2)
    torch.save({
        "model_state_dict": model.state_dict(),
        "checkpoint":       checkpoint,
        "model_name":       model_name,
        "held_out":         held_out,
        "train_domains":    train_domains,
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
    print("  Cross-Domain + In-Domain Subjectivity Scorer")
    print(f"  Models : {', '.join(MODELS.keys())}")
    print(f"  Domains: {', '.join(DOMAINS)}")
    print(f"  Mode   : {'INTERACTIVE' if INTERACTIVE else 'FULL DATASET'}")
    print("="*70)
    print("""
  For each held-out domain X:
    Train  : 5 domains combined (all except X)
    Test 1 : X test split                    ← cross-domain
    Test 2 : each of 5 training domains      ← in-domain per domain
    Test 3 : all 5 training domains combined ← in-domain combined
    Test 4 : all 6 domains combined          ← overall
    """)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Pre-load all datasets once ────────────
    print("Pre-loading all domain datasets...")
    all_datasets = {}
    for domain in DOMAINS:
        dataset = load_dataset("megagonlabs/subjqa", domain)
        if INTERACTIVE:
            for split in ["train", "validation", "test"]:
                n = min(INTERACTIVE_LIMIT, len(dataset[split]))
                dataset[split] = dataset[split].select(range(n))
        all_datasets[domain] = dataset
        print(f"  {domain}: train={len(dataset['train'])} | "
              f"val={len(dataset['validation'])} | test={len(dataset['test'])}")

    all_results = {model: {} for model in MODELS}

    for model_name, config in MODELS.items():
        print(f"\n{'═'*70}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'═'*70}")

        checkpoint = config["checkpoint"]
        batch_size = config["batch_size"]
        max_length = config["max_length"]
        tokenizer  = AutoTokenizer.from_pretrained(checkpoint)

        for held_out in DOMAINS:
            train_domains = [d for d in DOMAINS if d != held_out]

            print(f"\n{'─'*60}")
            print(f"  Held-out : {held_out.upper()}")
            print(f"  Train on : {', '.join(train_domains)}")
            print(f"{'─'*60}")

            # ── Build combined train/val from 5 domains ──
            train_parts = [
                SubjectivityDataset(
                    all_datasets[d]["train"], tokenizer, max_length
                ) for d in train_domains
            ]
            val_parts = [
                SubjectivityDataset(
                    all_datasets[d]["validation"], tokenizer, max_length
                ) for d in train_domains
            ]
            combined_train = ConcatDataset(train_parts)
            combined_val   = ConcatDataset(val_parts)

            print(f"  Combined train : {len(combined_train)} | "
                  f"Combined val: {len(combined_val)}")

            # ── Train on 5 domains ────────────
            trained_model, train_time = train_model(
                model_name, held_out,
                combined_train, combined_val,
                checkpoint, batch_size,
            )
            trained_model.eval()
            held_out_results = {"train_time_sec": train_time}

            # ── Test 1: held-out domain ───────
            print(f"\n  [Test 1] CROSS-DOMAIN: {held_out.upper()}")
            held_out_dataset = SubjectivityDataset(
                all_datasets[held_out]["test"], tokenizer, max_length
            )
            held_out_loader = DataLoader(held_out_dataset, batch_size=batch_size)
            cross_exp       = create_experiment(model_name, held_out, "eval_cross")
            cross_metrics   = evaluate_model(
                trained_model, held_out_loader, device,
                experiment=cross_exp, prefix="cross_"
            )
            cross_exp.end()
            held_out_results["cross_domain"] = cross_metrics
            print(f"    → RMSE: {cross_metrics['rmse']} | "
                  f"Pearson: {cross_metrics['pearson']}")

            # ── Test 2: each training domain ──
            print(f"\n  [Test 2] IN-DOMAIN (each of {train_domains}):")
            indomain_results    = {}
            indomain_all_datasets = []

            for train_d in train_domains:
                td_dataset = SubjectivityDataset(
                    all_datasets[train_d]["test"], tokenizer, max_length
                )
                td_loader  = DataLoader(td_dataset, batch_size=batch_size)
                td_exp     = create_experiment(
                    model_name, held_out, f"eval_indomain_{train_d}"
                )
                td_metrics = evaluate_model(
                    trained_model, td_loader, device,
                    experiment=td_exp, prefix=f"indomain_{train_d}_"
                )
                td_exp.end()
                indomain_results[train_d] = td_metrics
                indomain_all_datasets.append(td_dataset)
                print(f"    → {train_d:<16} RMSE: {td_metrics['rmse']} | "
                      f"Pearson: {td_metrics['pearson']}")

            held_out_results["in_domain_per_domain"]   = indomain_results
            held_out_results["avg_indomain_pearson"]   = round(float(np.mean(
                [indomain_results[d]["pearson"] for d in train_domains])), 4)
            held_out_results["avg_indomain_rmse"]      = round(float(np.mean(
                [indomain_results[d]["rmse"]    for d in train_domains])), 4)
            held_out_results["avg_indomain_spearman"]  = round(float(np.mean(
                [indomain_results[d]["spearman"] for d in train_domains])), 4)

            # ── Test 3: all training domains combined ──
            print(f"\n  [Test 3] IN-DOMAIN COMBINED (all 5 training domains):")
            combined_indomain        = ConcatDataset(indomain_all_datasets)
            combined_indomain_loader = DataLoader(combined_indomain, batch_size=batch_size)
            comb_indomain_exp        = create_experiment(
                model_name, held_out, "eval_indomain_combined"
            )
            comb_indomain_metrics = evaluate_model(
                trained_model, combined_indomain_loader, device,
                experiment=comb_indomain_exp, prefix="indomain_combined_"
            )
            comb_indomain_exp.end()
            held_out_results["in_domain_combined"] = comb_indomain_metrics
            print(f"    → RMSE: {comb_indomain_metrics['rmse']} | "
                  f"Pearson: {comb_indomain_metrics['pearson']}")

            # ── Test 4: all 6 domains combined ──
            print(f"\n  [Test 4] ALL DOMAINS COMBINED (6 domains):")
            all_test_datasets  = indomain_all_datasets + [held_out_dataset]
            all_combined       = ConcatDataset(all_test_datasets)
            all_combined_loader= DataLoader(all_combined, batch_size=batch_size)
            all_exp            = create_experiment(
                model_name, held_out, "eval_all_combined"
            )
            all_metrics = evaluate_model(
                trained_model, all_combined_loader, device,
                experiment=all_exp, prefix="all_combined_"
            )
            all_exp.end()
            held_out_results["all_combined"] = all_metrics
            print(f"    → RMSE: {all_metrics['rmse']} | "
                  f"Pearson: {all_metrics['pearson']}")

            all_results[model_name][held_out] = held_out_results

            del trained_model
            torch.cuda.empty_cache()

    # ── Summary tables ────────────────────────
    print("\n" + "="*80)
    print("  FINAL RESULTS SUMMARY")
    print("="*80)

    for model_name in MODELS:
        print(f"\n  Model: {model_name.upper()}")
        print(f"  {'Held-Out':<16} {'Cross':>10} {'Avg In-D':>10} "
              f"{'In-D Comb':>12} {'All Comb':>10}")
        print("  " + "─" * 62)
        for held_out in DOMAINS:
            res = all_results[model_name][held_out]
            print(
                f"  {held_out:<16}"
                f"{res['cross_domain']['pearson']:>10}"
                f"{res['avg_indomain_pearson']:>10}"
                f"{res['in_domain_combined']['pearson']:>12}"
                f"{res['all_combined']['pearson']:>10}"
            )

    # ── Best model overall ────────────────────
    print("\n" + "="*60)
    print("  BEST MODEL (by avg cross-domain Pearson)")
    print("="*60)
    model_avg = {}
    for model_name in MODELS:
        avg = np.mean([
            all_results[model_name][d]["cross_domain"]["pearson"]
            for d in DOMAINS
        ])
        model_avg[model_name] = round(float(avg), 4)
        print(f"  {model_name:<22} avg cross Pearson: {model_avg[model_name]}")

    best_model = max(model_avg, key=model_avg.get)
    print(f"\n  ★ Best model: {best_model} ({model_avg[best_model]})")

    # ── Save results ──────────────────────────
    results_path = f"{OUTPUT_DIR}/cross_domain_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── Log to CometML ────────────────────────
    summary_exp = Experiment(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
    )
    summary_exp.set_name("summary-cross-domain")
    summary_exp.add_tags(["summary", "cross-domain"])

    for model_name in MODELS:
        for held_out in DOMAINS:
            res = all_results[model_name][held_out]
            summary_exp.log_metrics({
                f"{model_name}_{held_out}_cross_pearson":        res["cross_domain"]["pearson"],
                f"{model_name}_{held_out}_cross_rmse":           res["cross_domain"]["rmse"],
                f"{model_name}_{held_out}_avg_indomain_pearson": res["avg_indomain_pearson"],
                f"{model_name}_{held_out}_indcomb_pearson":      res["in_domain_combined"]["pearson"],
                f"{model_name}_{held_out}_allcomb_pearson":      res["all_combined"]["pearson"],
            })
        summary_exp.log_metrics({
            f"{model_name}_avg_cross_pearson": model_avg[model_name]
        })
    summary_exp.end()
    print("Summary logged to CometML ✓")


if __name__ == "__main__":
    main()