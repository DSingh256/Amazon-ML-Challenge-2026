"""
S5b - The "LLM layer": a small multilingual language model that reads both records.

LightGBM judges a pair from ~70 numbers (similarities). A language model reads the two
records as text, so it can learn what the numbers miss: typos like "C0" / "Co", a name in
Hindi / Kannada / Telugu script next to its Latin spelling, re-ordered words, synonyms.
It is a "cross-encoder": both records go in together, one match score comes out
(Ditto: Li et al., "Deep Entity Matching with Pre-Trained Language Models", VLDB 2021).

  model  : XLM-RoBERTa base (MIT licence, 278M parameters, 100 languages), fine-tuned here
  learns : from the candidates of training Source 1 records that the LightGBM models never
           use (outside TRAIN_FRACTION), so its score is a fair feature for them
  scores : only the pairs stage 1 is unsure about (a GPU cannot read all ~26M test pairs in
           time); every other pair has no score (NaN), which LightGBM handles
  output : <work>/<split>/ce_scores.parquet  (s1_row, s23_row, ce = log-odds of a match)

Stage 2 (train.py / assign.py) then gets two more features: ce and ce_others_max (the
best score of the other candidates of the same Source 1 record).
Runs in a Kaggle GPU notebook (run_pipeline.py llm); on a CPU only for small tests.
torch / transformers are imported inside the functions: train and assign do not need them.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import CROSS_ENCODER, HOLDOUT_PERCENT, SEED
from features import training_records
from io_utils import id_bucket, log, read_ground_truth_pairs, truth_as_rows

CE_COLUMNS = ["ce", "ce_others_max"]
SCORE_FILE = "ce_scores.parquet"


# ---------------------------------------------------------------------------
# Which pairs get a score: the ones stage 1 is unsure about
# ---------------------------------------------------------------------------
def stage1_probabilities(split_dir, model_dir, split):
    """Stage-1 probability of every pair of features.parquet, exactly as train.py / assign.py
    give it to stage 2: test and holdout pairs get the mean of both stage-1 models, the
    other training pairs the model that did not learn from their record (cross-fitting).
    Returns a DataFrame s1_row, s23_row, p1 in file order."""
    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    models = [lgb.Booster(model_file=str(Path(model_dir) / name)) for name in meta["stage1_models"]]
    columns = meta["stage1_features"]
    if split == "train":
        assert len(models) == 2, "cross-fitting needs the two stage-1 models"
        raw = id_bucket(pd.read_parquet(Path(split_dir) / "s1_clean.parquet", columns=["entity_id"])["entity_id"])
        fold, holdout = raw % 2, raw % 100 < HOLDOUT_PERCENT
    parts = []
    file = pq.ParquetFile(Path(split_dir) / "features.parquet")
    for batch in file.iter_batches(batch_size=1_000_000, columns=["s1_row", "s23_row"] + columns):
        batch = batch.to_pandas()
        predictions = [m.predict(batch[columns]) for m in models]
        p = sum(predictions) / len(predictions)
        if split == "train":
            row = batch["s1_row"].to_numpy()
            # model k learned from fold k, so a fold-0 record gets model 1 and the other way round
            p = np.where(holdout[row], p, np.where(fold[row] == 0, predictions[1], predictions[0]))
        parts.append(pd.DataFrame({"s1_row": batch["s1_row"].to_numpy(), "s23_row": batch["s23_row"].to_numpy(),
                                   "p1": p.astype(np.float32)}))
    return pd.concat(parts, ignore_index=True)


def unsure_limit(p1, config=CROSS_ENCODER):
    """A pair is "unsure" when |p1 - 0.5| < limit. The limit is config.max_certainty, made
    smaller when that would give more than config.score_budget pairs (decided on the test
    pairs, then used for the training pairs too)."""
    certainty = np.abs(p1 - 0.5)
    limit = config.max_certainty
    if (certainty < limit).sum() > config.score_budget:
        limit = float(np.partition(certainty, config.score_budget)[config.score_budget])
    return limit


# ---------------------------------------------------------------------------
# Text of a pair
# ---------------------------------------------------------------------------
def record_texts(table):
    """'name | address' of every row (original text: the model reads every script)."""
    name = table["business_name"].fillna("").astype(str)
    address = table["business_address"].fillna("").astype(str)
    return (name + " | " + address).str.strip(" |").to_numpy(dtype=object)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class PairModel:
    """Tokenizer + cross-encoder with one output (log-odds of a match), on the GPU if there is one."""

    def __init__(self, config=CROSS_ENCODER):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.config = torch, config
        torch.manual_seed(SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model, num_labels=1).to(self.device)
        self.n_gpus = torch.cuda.device_count()
        self.parallel = torch.nn.DataParallel(self.model) if self.n_gpus > 1 else self.model
        log(f"cross-encoder {config.model} on {self.device} ({self.n_gpus} GPU), "
            f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M parameters")

    def encode(self, left, right):
        batch = self.tokenizer(list(left), list(right), truncation=True, max_length=self.config.max_tokens,
                               padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def logits(self, left, right):
        with self.torch.autocast(self.device.type, dtype=self.torch.float16, enabled=self.device.type == "cuda"):
            return self.parallel(**self.encode(left, right)).logits.float().squeeze(-1)

    def fine_tune(self, left, right, labels):
        """A few passes over the pairs (binary cross-entropy), linear warm-up and decay."""
        from transformers import get_linear_schedule_with_warmup
        torch, config = self.torch, self.config
        size = config.batch_size
        steps = config.epochs * -(-len(labels) // size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        loss_function = torch.nn.BCEWithLogitsLoss()
        labels = torch.tensor(labels, dtype=torch.float32)
        rng = np.random.default_rng(SEED)
        self.parallel.train()
        step, start_time, running = 0, time.time(), 0.0
        for epoch in range(config.epochs):
            order = rng.permutation(len(labels))
            for start in range(0, len(order), size):
                rows = order[start:start + size]
                loss = loss_function(self.logits(left[rows], right[rows]), labels[rows].to(self.device))
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
                step += 1
                running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
                if step % 500 == 0 or step == steps:
                    log(f"  epoch {epoch + 1}, step {step:,}/{steps:,}: loss {running:.4f} "
                        f"({step * size / (time.time() - start_time):,.0f} pairs/s)")
        self.parallel.eval()

    def score(self, left, right, chunk=50_000):
        """Log-odds of every pair. Pairs of similar length are batched together (less padding)."""
        torch = self.torch
        out = np.empty(len(left), dtype=np.float32)
        size = self.config.infer_batch
        start_time = time.time()
        with torch.inference_mode():
            for lo in range(0, len(left), chunk):
                hi = min(lo + chunk, len(left))
                lengths = np.array([len(a) + len(b) for a, b in zip(left[lo:hi], right[lo:hi])])
                order = lo + np.argsort(lengths, kind="stable")
                for start in range(0, len(order), size):
                    rows = order[start:start + size]
                    out[rows] = self.logits(left[rows], right[rows]).cpu().numpy()
                if (lo // chunk) % 20 == 0 or hi == len(left):
                    log(f"  scored {hi:,}/{len(left):,} pairs ({hi / (time.time() - start_time):,.0f} pairs/s)")
        return out


# ---------------------------------------------------------------------------
# Features for stage 2
# ---------------------------------------------------------------------------
def ce_features(s1_row, s23_row, scores):
    """ce and ce_others_max of every pair (s1_row, s23_row), in the input order.
    scores: DataFrame s1_row, s23_row, ce (pairs without a score get NaN)."""
    keys = pd.DataFrame({"s1_row": np.asarray(s1_row), "s23_row": np.asarray(s23_row)})
    ce = keys.merge(scores[["s1_row", "s23_row", "ce"]], on=["s1_row", "s23_row"], how="left")["ce"]
    ce = pd.Series(ce.to_numpy(dtype=np.float32))
    record = keys["s1_row"].to_numpy()
    best = ce.groupby(record).transform("max")
    is_best = ce == best
    n_best = is_best.groupby(record).transform("sum")
    second = ce.where(~is_best).groupby(record).transform("max")
    others = np.where(is_best, np.where(n_best > 1, best, second), best)
    return pd.DataFrame({"ce": ce.to_numpy(), "ce_others_max": others.astype(np.float32)})


# ---------------------------------------------------------------------------
# Whole step
# ---------------------------------------------------------------------------
def learning_pairs(train_dir, data_dir, s1, s23, config=CROSS_ENCODER):
    """The candidates of config.train_records random training Source 1 records that the
    LightGBM models never see, with labels. Only the hard ones are kept: the first
    config.max_learning_rank candidates of each record (+ every true match), because the far
    candidates are easy non-matches that stage 1 already gets right.
    Returns (DataFrame s1_row, s23_row, label)."""
    info = json.loads((train_dir / "features_info.json").read_text())
    unused = np.flatnonzero(~training_records(s1, info["train_fraction"]))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(unused, size=min(config.train_records, len(unused)), replace=False)
    candidates = pd.read_parquet(train_dir / "candidates.parquet", columns=["s1_row", "s23_row", "candidate_rank"])
    pairs = candidates[np.isin(candidates["s1_row"].to_numpy(), chosen)].reset_index(drop=True)
    del candidates
    truth = truth_as_rows(read_ground_truth_pairs(data_dir), s1, s23)
    truth["label"] = 1
    pairs = pairs.merge(truth, on=["s1_row", "s23_row"], how="left")
    pairs["label"] = pairs["label"].fillna(0).astype(np.int8)
    hard = (pairs["candidate_rank"].to_numpy() <= config.max_learning_rank) | (pairs["label"].to_numpy() == 1)
    pairs = pairs.loc[hard, ["s1_row", "s23_row", "label"]].reset_index(drop=True)
    log(f"cross-encoder learns from {len(pairs):,} pairs of {len(chosen):,} unused Source 1 records "
        f"({pairs['label'].mean():.1%} matches)")
    return pairs


def auc_of(labels, scores):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(scores)
    return roc_auc_score(labels[ok], scores[ok]) if len(np.unique(labels[ok])) == 2 else float("nan")


def run_cross_encoder(work_dir, data_dir, config=CROSS_ENCODER):
    """Fine-tune on unused training records, then score the unsure pairs of train and test."""
    work_dir = Path(work_dir)
    train_dir, test_dir, model_dir = work_dir / "train", work_dir / "test", work_dir / "models"

    # 1. which pairs: decided on the test pairs, same limit for the training pairs
    test_p1 = stage1_probabilities(test_dir, model_dir, "test")
    limit = unsure_limit(test_p1["p1"].to_numpy(), config)
    train_p1 = stage1_probabilities(train_dir, model_dir, "train")
    todo = {}
    for split, table in (("test", test_p1), ("train", train_p1)):
        unsure = np.abs(table["p1"].to_numpy() - 0.5) < limit
        todo[split] = table[unsure].reset_index(drop=True)
        log(f"{split}: {unsure.sum():,} of {len(table):,} pairs are unsure "
            f"(stage-1 probability {0.5 - limit:.4f} .. {0.5 + limit:.4f})")
    del test_p1, train_p1

    # 2. learn
    columns = ["business_name", "business_address"]
    s1 = pd.read_parquet(train_dir / "s1_clean.parquet", columns=["entity_id"] + columns)
    s23 = pd.read_parquet(train_dir / "s23_clean.parquet", columns=["entity_id"] + columns)
    left_text, right_text = record_texts(s1), record_texts(s23)
    pairs = learning_pairs(train_dir, data_dir, s1, s23, config)
    rng = np.random.default_rng(SEED + 1)
    records = pairs["s1_row"].unique()
    check_records = rng.choice(records, size=max(1, len(records) // 20), replace=False)
    is_check = np.isin(pairs["s1_row"].to_numpy(), check_records)
    learn, check = pairs[~is_check], pairs[is_check]
    model = PairModel(config)
    model.fine_tune(left_text[learn["s1_row"]], right_text[learn["s23_row"]], learn["label"].to_numpy())
    check_scores = model.score(left_text[check["s1_row"]], right_text[check["s23_row"]])
    log(f"cross-encoder AUC on {len(check):,} pairs of unseen records: {auc_of(check['label'].to_numpy(), check_scores):.5f}")
    try:    # kept, so a later run can score more pairs without learning again
        model.model.save_pretrained(model_dir / "cross_encoder")
        model.tokenizer.save_pretrained(model_dir / "cross_encoder")
    except Exception as error:
        log(f"could not save the cross-encoder: {error}")

    # 3. score the unsure training pairs (stage 2 learns from them) ...
    unsure = todo["train"]
    unsure["ce"] = model.score(left_text[unsure["s1_row"]], right_text[unsure["s23_row"]])
    unsure[["s1_row", "s23_row", "ce"]].to_parquet(train_dir / SCORE_FILE, index=False)
    labels = pd.read_parquet(train_dir / "features.parquet", columns=["s1_row", "s23_row", "label"])
    labelled = unsure.merge(labels, on=["s1_row", "s23_row"])
    log(f"unsure training pairs: AUC stage 1 {auc_of(labelled['label'].to_numpy(), labelled['p1'].to_numpy()):.4f} | "
        f"cross-encoder {auc_of(labelled['label'].to_numpy(), labelled['ce'].to_numpy()):.4f}")
    del s1, s23, left_text, right_text, labels, labelled

    # 4. ... and the unsure test pairs
    s1 = pd.read_parquet(test_dir / "s1_clean.parquet", columns=columns)
    s23 = pd.read_parquet(test_dir / "s23_clean.parquet", columns=columns)
    left_text, right_text = record_texts(s1), record_texts(s23)
    unsure = todo["test"]
    unsure["ce"] = model.score(left_text[unsure["s1_row"]], right_text[unsure["s23_row"]])
    unsure[["s1_row", "s23_row", "ce"]].to_parquet(test_dir / SCORE_FILE, index=False)
    log(f"saved {train_dir / SCORE_FILE} and {test_dir / SCORE_FILE}")
