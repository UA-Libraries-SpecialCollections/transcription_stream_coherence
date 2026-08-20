#!/usr/bin/env python3
"""
Language-Likeness Evaluator (Python 3.10)

A Tkinter GUI application that:
- Evaluates plain text files for language-likeness signals
- Writes one JSON report per input file
- Loads JSON reports and visualizes them as radar charts (single or overlay)

Key metrics:
- Language model: cross-entropy / perplexity / log-likelihood (Hugging Face causal LM)
- OOV ratio using an English dictionary resource (wordfreq)
- Character class ratios: alpha / digit / punctuation / control
- Repeated character runs as percent of characters
- Characters vs. whitespace ratios

Dependencies (install via pip):
    pip install matplotlib numpy wordfreq transformers torch

Notes:
- The default LM ("distilgpt2") will download on first run (cached thereafter).
- For large texts, evaluation uses a sliding-window NLL computation.

"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import random
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Matplotlib (TkAgg backend)
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----------------------------
# Configuration / Data classes
# ----------------------------

@dataclass
class AnalyzerConfig:
    model_name: str = "distilgpt2"
    device: str = "auto"            # "auto" | "cpu" | "cuda"
    max_chars: int = 200_000        # cap text analyzed (performance control)
    min_repeat_run: int = 3         # repeated character run threshold
    repeat_exclude_whitespace: bool = True

    # Radar normalization parameters
    # (Used to convert LM perplexity into 0..1 "quality" scores)
    log_ppl_good: float = 3.5       # ppl ~ 33
    log_ppl_bad: float = 6.0        # ppl ~ 403
    # For a raw-scale radar axis, clip log perplexity to 0..log_ppl_clip then scale to 0..1
    log_ppl_clip: float = 8.0       # ppl ~ 2981

    # Whitespace "quality" shaping: best around whitespace_target, degrades linearly to 0 at target±whitespace_range
    whitespace_target: float = 0.18
    whitespace_range: float = 0.18

    # LM window-distribution scaling (Option B)
    # Used for the LM consistency spoke (std-dev of window losses).
    lm_window_std_good: float = 0.12
    lm_window_std_bad: float = 0.60

    # Lexical sketch / overlap support
    lexical_remove_stopwords: bool = True
    lexical_min_token_len: int = 2

    # MinHash sketch for estimating lexical overlap across items (Jaccard)
    minhash_k: int = 64
    minhash_seed: int = 1


@dataclass
class FileMetadata:
    path: str
    basename: str
    bytes: int
    sha256: str
    modified_time_utc: str


# ----------------------------
# Utility functions
# ----------------------------

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# A compact built-in English stopword list (no extra dependency).
# Purpose: reduce the impact of function words on lexical overlap signals.
_EN_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each",
    "few","for","from","further",
    "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's",
    "hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's",
    "me","more","most","mustn't","my","myself",
    "no","nor","not",
    "of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they",
    "they'd","they'll","they're","they've","this","those","through","to","too",
    "under","until","up",
    "very",
    "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's",
    "where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
    "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves",
}

_MINHASH_PRIME = (1 << 61) - 1  # a large prime (Mersenne prime), fits safely in signed 64-bit

def _stable_hash64(s: str) -> int:
    """Deterministic 64-bit hash for strings (stable across Python runs)."""
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()[:8]
    return int.from_bytes(h, byteorder="big", signed=False)

class MinHasher:
    """Simple MinHash sketch for estimating Jaccard similarity between token sets."""
    def __init__(self, k: int = 64, seed: int = 1, prime: int = _MINHASH_PRIME) -> None:
        self.k = int(k)
        self.seed = int(seed)
        self.prime = int(prime)
        rnd = random.Random(self.seed)
        # Pre-generate k hash function parameters (a, b)
        # h_i(x) = (a_i * x + b_i) mod prime
        self._ab: List[Tuple[int, int]] = []
        for _ in range(self.k):
            a = rnd.randrange(1, self.prime - 1)
            b = rnd.randrange(0, self.prime - 1)
            self._ab.append((a, b))

    def signature(self, token_set: set[str]) -> List[int]:
        if not token_set:
            return []
        sig = [self.prime] * self.k
        for tok in token_set:
            x = _stable_hash64(tok) % self.prime
            for i, (a, b) in enumerate(self._ab):
                hv = (a * x + b) % self.prime
                if hv < sig[i]:
                    sig[i] = hv
        return sig

def minhash_jaccard_estimate(sig_a_hex: Optional[List[str]], sig_b_hex: Optional[List[str]]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures stored as hex strings."""
    if not sig_a_hex or not sig_b_hex:
        return float("nan")
    if len(sig_a_hex) != len(sig_b_hex) or len(sig_a_hex) == 0:
        return float("nan")
    matches = 0
    for a, b in zip(sig_a_hex, sig_b_hex):
        if a == b:
            matches += 1
    return matches / float(len(sig_a_hex))

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def safe_read_text_file(path: Path) -> Tuple[bytes, str]:
    """
    Reads a file as bytes, then decodes as UTF-8 with replacement.
    Returns (raw_bytes, decoded_text).
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return raw, text

def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def unicode_category(c: str) -> str:
    try:
        return unicodedata.category(c)
    except Exception:
        return "Cn"

def normalize_for_language_model(text: str) -> str:
    """
    Simple normalization to reduce pathological effects on tokenization/LM scoring.
    Keeps content, collapses whitespace, removes NULs.
    """
    # Remove NULs explicitly
    text = text.replace("\x00", " ")
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# Batch visualization helpers
# ----------------------------

_NAT_SORT_RE = re.compile(r"(\d+)")

def natural_sort_key(s: str) -> List[Any]:
    """Natural sort key that orders 'item2' before 'item10'."""
    parts = _NAT_SORT_RE.split(s)
    out: List[Any] = []
    for p in parts:
        if p.isdigit():
            try:
                out.append(int(p))
            except Exception:
                out.append(p)
        else:
            out.append(p.lower())
    return out

def filename_pattern_sort_key(name: str, pattern: str) -> Tuple[int, int, Any, List[Any]]:
    """
    Sort key that optionally extracts a primary ordering token from `name` using regex `pattern`.

    - If `pattern` matches and an integer group can be parsed, that integer is used for ordering.
    - Otherwise falls back to natural sort.

    Returns a tuple that is always comparable.
    """
    if pattern:
        try:
            m = re.search(pattern, name)
            if m:
                g = m.group(1) if m.lastindex else m.group(0)
                try:
                    val = int(g)
                    return (0, 0, val, natural_sort_key(name))
                except Exception:
                    return (0, 1, str(g).lower(), natural_sort_key(name))
        except re.error:
            # invalid pattern -> ignore and fall back
            pass
    return (1, 0, 0, natural_sort_key(name))

def rolling_median_1d(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median with edge padding. Returns an array of same length as y."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y.copy()
    if window <= 1:
        return y.copy()

    w = int(window)
    if w < 1:
        w = 1
    if w % 2 == 0:
        w += 1  # enforce odd window for centering

    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")

    # Prefer a vectorized implementation
    try:
        sw = np.lib.stride_tricks.sliding_window_view(ypad, w)
        med = np.nanmedian(sw, axis=-1)
        return med.astype(float)
    except Exception:
        # Fallback: simple loop (OK for <= ~10k items and modest windows)
        out = np.empty_like(y, dtype=float)
        for i in range(n):
            lo = max(0, i - pad)
            hi = min(n, i + pad + 1)
            out[i] = float(np.nanmedian(y[lo:hi]))
        return out


# ----------------------------
# Dictionary (OOV) scorer
# ----------------------------

class EnglishDictionary:
    """
    Lightweight dictionary interface.

    Uses `wordfreq` as a dictionary-like resource: words with non-zero zipf_frequency
    are treated as in-vocabulary.
    """
    def __init__(self) -> None:
        try:
            import wordfreq  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'wordfreq'. Install with: pip install wordfreq"
            ) from e
        self._wordfreq = wordfreq

    def is_in_vocab(self, word: str) -> bool:
        w = word.lower()
        # wordfreq.zipf_frequency returns 0.0 for unknown words (common behavior),
        # but we keep the condition loose to avoid false OOV for rare forms.
        try:
            return float(self._wordfreq.zipf_frequency(w, "en")) > 0.0
        except Exception:
            return False


# ----------------------------
# Language model scorer
# ----------------------------

class CausalLMScorer:
    """
    Hugging Face causal LM scorer that computes:
    - average cross-entropy (nats/token)
    - perplexity
    - total / average log-likelihood

    Implements a sliding-window evaluation so it works on long text.
    """
    def __init__(self, model_name: str = "distilgpt2", device: str = "auto") -> None:
        self.model_name = model_name
        self.device_pref = device
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependencies for language model scoring.\n"
                "Install with: pip install transformers torch"
            ) from e

        self._torch = torch
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.eval()

        # Device selection
        device = "cpu"
        if self.device_pref == "cuda":
            device = "cuda"
        elif self.device_pref == "cpu":
            device = "cpu"
        else:
            # auto
            if torch.cuda.is_available():
                device = "cuda"
        model.to(device)

        self._tokenizer = tok
        self._model = model
        self._device = device
        self._loaded = True

    def score(self, text: str) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        device = self._device

        text = normalize_for_language_model(text)
        if not text:
            return {
                "model_name": self.model_name,
                "device": device,
                "token_count": 0,
                "avg_cross_entropy_nats": None,
                "avg_cross_entropy_bits": None,
                "perplexity": None,
                "avg_log_likelihood": None,
                "total_log_likelihood": None,
                "log_perplexity": None,
                # Window-distribution stats (Option B)
                "window_count": 0,
                "window_loss_mean_nats": None,
                "window_loss_std_nats": None,
                "window_loss_p10_nats": None,
                "window_loss_p90_nats": None,
                "window_loss_min_nats": None,
                "window_loss_max_nats": None,
            }

        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"][0].to(device)

        # Sliding window parameters
        # Prefer model.config.n_positions (GPT-2 uses 1024), but fall back to tokenizer.model_max_length
        max_length = getattr(model.config, "n_positions", None)
        if not isinstance(max_length, int) or max_length <= 0:
            max_length = int(getattr(tokenizer, "model_max_length", 1024))
        max_length = min(max_length, 2048)  # guardrail for some tokenizers that report huge max length

        # If extremely short, do one pass (not enough context to compute a meaningful loss)
        if input_ids.numel() < 2:
            return {
                "model_name": self.model_name,
                "device": device,
                "token_count": int(max(0, input_ids.numel() - 1)),
                "avg_cross_entropy_nats": None,
                "avg_cross_entropy_bits": None,
                "perplexity": None,
                "avg_log_likelihood": None,
                "total_log_likelihood": None,
                "log_perplexity": None,
                # Window-distribution stats (Option B)
                "window_count": 0,
                "window_loss_mean_nats": None,
                "window_loss_std_nats": None,
                "window_loss_p10_nats": None,
                "window_loss_p90_nats": None,
                "window_loss_min_nats": None,
                "window_loss_max_nats": None,
            }

        stride = max_length // 2
        if stride < 128:
            stride = max(1, max_length - 1)

        nll_sum = 0.0
        token_count = 0

        # Track per-window mean losses (nats/token) so we can compute:
        # - p10/p90 loss: "best/worst chunk"
        # - std-dev: stability/consistency
        window_losses: List[float] = []

        with torch.no_grad():
            for i in range(0, input_ids.size(0), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, input_ids.size(0))
                trg_len = end_loc - i  # how many tokens are predicted in this window

                input_ids_slice = input_ids[begin_loc:end_loc]
                labels = input_ids_slice.clone()

                # mask tokens we don't want to predict
                labels[: -trg_len] = -100

                outputs = model(input_ids_slice.unsqueeze(0), labels=labels.unsqueeze(0))
                # outputs.loss is mean NLL over trg_len tokens (nats/token)
                win_loss = float(outputs.loss)
                nll = win_loss * float(trg_len)

                window_losses.append(win_loss)
                nll_sum += nll
                token_count += int(trg_len)

                if end_loc == input_ids.size(0):
                    break

        if token_count <= 0:
            return {
                "model_name": self.model_name,
                "device": device,
                "token_count": 0,
                "avg_cross_entropy_nats": None,
                "avg_cross_entropy_bits": None,
                "perplexity": None,
                "avg_log_likelihood": None,
                "total_log_likelihood": None,
                "log_perplexity": None,
                # Window-distribution stats (Option B)
                "window_count": 0,
                "window_loss_mean_nats": None,
                "window_loss_std_nats": None,
                "window_loss_p10_nats": None,
                "window_loss_p90_nats": None,
                "window_loss_min_nats": None,
                "window_loss_max_nats": None,
            }

        avg_ce_nats = nll_sum / token_count
        ppl = float(math.exp(avg_ce_nats)) if avg_ce_nats < 50 else float("inf")
        log_ppl = float(math.log(ppl)) if math.isfinite(ppl) and ppl > 0 else float("inf")
        avg_ll = -avg_ce_nats
        total_ll = -nll_sum

        # ---- window-distribution stats ----
        def _percentile(vals: List[float], q: float) -> Optional[float]:
            if not vals:
                return None
            if q <= 0.0:
                return float(min(vals))
            if q >= 1.0:
                return float(max(vals))
            xs = sorted(float(v) for v in vals)
            if len(xs) == 1:
                return xs[0]
            pos = (len(xs) - 1) * q
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return xs[lo]
            frac = pos - lo
            return xs[lo] * (1.0 - frac) + xs[hi] * frac

        window_count = len(window_losses)
        if window_count > 0:
            win_mean = float(sum(window_losses) / window_count)
            if window_count >= 2:
                var = sum((x - win_mean) ** 2 for x in window_losses) / window_count
                win_std = float(math.sqrt(var))
            else:
                win_std = 0.0

            win_p10 = _percentile(window_losses, 0.10)
            win_p90 = _percentile(window_losses, 0.90)
            win_min = float(min(window_losses))
            win_max = float(max(window_losses))
        else:
            win_mean = None
            win_std = None
            win_p10 = None
            win_p90 = None
            win_min = None
            win_max = None

        return {
            "model_name": self.model_name,
            "device": device,
            "token_count": token_count,
            "avg_cross_entropy_nats": avg_ce_nats,
            "avg_cross_entropy_bits": avg_ce_nats / math.log(2),
            "perplexity": ppl,
            "avg_log_likelihood": avg_ll,
            "total_log_likelihood": total_ll,
            "log_perplexity": log_ppl,
            # Window-distribution stats (Option B)
            "window_count": window_count,
            "window_loss_mean_nats": win_mean,
            "window_loss_std_nats": win_std,
            "window_loss_p10_nats": win_p10,
            "window_loss_p90_nats": win_p90,
            "window_loss_min_nats": win_min,
            "window_loss_max_nats": win_max,
        }



# ----------------------------
# Core text analysis
# ----------------------------

class TextAnalyzer:
    def __init__(self, cfg: AnalyzerConfig) -> None:
        self.cfg = cfg
        self._dict = EnglishDictionary()
        self._lm = CausalLMScorer(model_name=cfg.model_name, device=cfg.device)
        self._minhasher = MinHasher(k=cfg.minhash_k, seed=cfg.minhash_seed)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        cfg = self.cfg
        analyzed_text = text[: cfg.max_chars] if cfg.max_chars > 0 else text
        truncated = (len(analyzed_text) != len(text))

        # Character statistics
        total = len(analyzed_text)
        alpha = digit = punct = control = whitespace = other = 0

        for ch in analyzed_text:
            if ch.isspace():
                whitespace += 1
                continue

            cat = unicode_category(ch)
            if ch.isalpha():
                alpha += 1
            elif ch.isdigit():
                digit += 1
            elif cat.startswith("P"):
                punct += 1
            elif cat.startswith("C"):
                control += 1
            else:
                other += 1

        non_ws = total - whitespace
        char_to_ws_ratio = (non_ws / whitespace) if whitespace > 0 else float("inf")

        def ratio(n: int) -> float:
            return (n / total) if total > 0 else 0.0

        # Repeated character runs
        rep_all, rep_nonws, max_run = self._repeated_runs(analyzed_text)

        # Tokenization for dictionary
        tokens = _WORD_RE.findall(analyzed_text)
        token_count = len(tokens)
        oov = 0
        if token_count > 0:
            for w in tokens:
                if not self._dict.is_in_vocab(w):
                    oov += 1
            oov_ratio = oov / token_count
        else:
            oov_ratio = 1.0  # no recognizable words -> treat as fully OOV for language-likeness baselines

        # Lexical metrics and MinHash sketch (supports estimating lexical overlap across items)
        tokens_lc = [t.lower() for t in tokens if len(t) >= int(cfg.lexical_min_token_len)]
        unique_tokens = set(tokens_lc)

        type_token_ratio = (len(unique_tokens) / len(tokens_lc)) if tokens_lc else 0.0
        repeat_token_ratio = (1.0 - type_token_ratio) if tokens_lc else 0.0

        if bool(cfg.lexical_remove_stopwords):
            content_tokens_lc = [t for t in tokens_lc if t not in _EN_STOPWORDS]
        else:
            content_tokens_lc = tokens_lc

        content_unique = set(content_tokens_lc)
        content_type_token_ratio = (len(content_unique) / len(content_tokens_lc)) if content_tokens_lc else 0.0
        content_repeat_token_ratio = (1.0 - content_type_token_ratio) if content_tokens_lc else 0.0

        minhash_sig_hex: Optional[List[str]] = None
        if content_unique:
            sig_int = self._minhasher.signature(content_unique)
            if sig_int:
                minhash_sig_hex = [f"{v:016x}" for v in sig_int]

        lexical = {
            "tokenization": {
                "pattern": _WORD_RE.pattern,
                "lowercase": True,
                "min_token_len": int(cfg.lexical_min_token_len),
                "stopwords_removed": bool(cfg.lexical_remove_stopwords),
            },
            "tokens": {
                "token_count": int(len(tokens_lc)),
                "unique_token_count": int(len(unique_tokens)),
                "type_token_ratio": float(type_token_ratio),
                "repeat_token_ratio": float(repeat_token_ratio),
            },
            "content_tokens": {
                "token_count": int(len(content_tokens_lc)),
                "unique_token_count": int(len(content_unique)),
                "type_token_ratio": float(content_type_token_ratio),
                "repeat_token_ratio": float(content_repeat_token_ratio),
            },
            "minhash": {
                "k": int(cfg.minhash_k),
                "seed": int(cfg.minhash_seed),
                "prime": int(self._minhasher.prime),
                "signature_hex": minhash_sig_hex,
            },
        }


        # Language model scoring
        lm_scores = self._lm.score(analyzed_text)

        # Raw features (mostly 0..1, except LM and char_to_ws_ratio)
        features_raw: Dict[str, Any] = {
            "alpha_ratio": ratio(alpha),
            "digit_ratio": ratio(digit),
            "punct_ratio": ratio(punct),
            "control_ratio": ratio(control),
            "other_ratio": ratio(other),
            "whitespace_ratio": ratio(whitespace),
            "non_whitespace_ratio": ratio(non_ws),
            "char_to_whitespace_ratio": char_to_ws_ratio,
            "repeat_run_char_ratio": rep_all,
            "repeat_run_nonws_char_ratio": rep_nonws,
            "max_repeat_run": max_run,
            "oov_ratio": oov_ratio,
            "type_token_ratio": type_token_ratio,
            "repeat_token_ratio": repeat_token_ratio,
            "content_type_token_ratio": content_type_token_ratio,
            "content_repeat_token_ratio": content_repeat_token_ratio,
        }

        # Derived / scaled features for radar
        derived = self._derive_radar_values(features_raw, lm_scores)

        return {
            "analyzed_chars": total,
            "truncated_to_max_chars": truncated,
            "char_counts": {
                "total": total,
                "alpha": alpha,
                "digit": digit,
                "punct": punct,
                "control": control,
                "whitespace": whitespace,
                "other": other,
                "non_whitespace": non_ws,
            },
            "dictionary": {
                "token_count": token_count,
                "oov_count": oov,
                "oov_ratio": oov_ratio,
            },
            "lexical": lexical,
            "language_model": lm_scores,
            "features_raw": features_raw,
            "radar": derived,
        }

    def analyze_file(self, path: Path) -> Dict[str, Any]:
        raw, text = safe_read_text_file(path)

        st = path.stat()
        meta = FileMetadata(
            path=str(path),
            basename=path.name,
            bytes=len(raw),
            sha256=sha256_hex(raw),
            modified_time_utc=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        )

        analysis = self.analyze_text(text)

        report: Dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": utc_now_iso(),
            "config": asdict(self.cfg),
            "file": asdict(meta),
            "analysis": analysis,
        }
        return report

    def _repeated_runs(self, text: str) -> Tuple[float, float, int]:
        """
        Returns:
            (repeat_run_char_ratio_all, repeat_run_char_ratio_nonws, max_run_len)
        where repeat_run_* is the fraction of characters that belong to runs
        of length >= min_repeat_run.
        """
        cfg = self.cfg
        n = len(text)
        if n == 0:
            return 0.0, 0.0, 0

        rep_all = 0
        rep_nonws = 0
        max_run = 1

        i = 0
        while i < n:
            ch = text[i]
            j = i + 1
            while j < n and text[j] == ch:
                j += 1
            run_len = j - i
            if run_len > max_run:
                max_run = run_len

            if run_len >= cfg.min_repeat_run:
                rep_all += run_len
                if not ch.isspace():
                    rep_nonws += run_len

            i = j

        rep_all_ratio = rep_all / n
        nonws_count = sum(1 for c in text if not c.isspace())
        rep_nonws_ratio = (rep_nonws / nonws_count) if nonws_count > 0 else 0.0

        # Optionally exclude whitespace runs for the "all" ratio (common OCR artifact: long spaces)
        if cfg.repeat_exclude_whitespace:
            # recompute rep_all excluding whitespace
            rep_all_excl_ws = 0
            i = 0
            while i < n:
                ch = text[i]
                j = i + 1
                while j < n and text[j] == ch:
                    j += 1
                run_len = j - i
                if run_len >= cfg.min_repeat_run and (not ch.isspace()):
                    rep_all_excl_ws += run_len
                i = j
            rep_all_ratio = rep_all_excl_ws / n

        return rep_all_ratio, rep_nonws_ratio, max_run

    def _derive_radar_values(self, raw: Dict[str, Any], lm: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates two radar-ready vectors:
        - "quality": higher is better language-likeness
        - "raw_scaled": mostly raw 0..1 ratios + scaled LM axis (not all axes mean "better when higher")
        """
        cfg = self.cfg

        log_ppl = lm.get("log_perplexity", None)
        if log_ppl is None or (isinstance(log_ppl, float) and not math.isfinite(log_ppl)):
            lm_quality = 0.0
            lm_log_ppl_scaled = 1.0
        else:
            # Quality: 1 when <= good, 0 when >= bad, linear in between
            lm_quality = clamp01((cfg.log_ppl_bad - float(log_ppl)) / (cfg.log_ppl_bad - cfg.log_ppl_good))
            # Raw scaled for plotting: 0..1 increasing with log perplexity
            lm_log_ppl_scaled = clamp01(float(log_ppl) / cfg.log_ppl_clip)

        oov_ratio = float(raw.get("oov_ratio", 1.0))
        whitespace_ratio = float(raw.get("whitespace_ratio", 0.0))

        # Whitespace "quality": best around target, decreases linearly away from target.
        if cfg.whitespace_range <= 0:
            whitespace_quality = 0.0
        else:
            whitespace_quality = 1.0 - (abs(whitespace_ratio - cfg.whitespace_target) / cfg.whitespace_range)
            whitespace_quality = clamp01(whitespace_quality)

        quality_axes = [
            "lm_quality",
            "vocab_quality",
            "alpha_ratio",
            "no_digits",
            "no_punct",
            "no_control",
            "no_repeats",
            "whitespace_balance",
        ]
        quality_values = {
            "lm_quality": lm_quality,
            "vocab_quality": clamp01(1.0 - oov_ratio),
            "alpha_ratio": clamp01(float(raw.get("alpha_ratio", 0.0))),
            "no_digits": clamp01(1.0 - float(raw.get("digit_ratio", 0.0))),
            "no_punct": clamp01(1.0 - float(raw.get("punct_ratio", 0.0))),
            "no_control": clamp01(1.0 - float(raw.get("control_ratio", 0.0))),
            "no_repeats": clamp01(1.0 - float(raw.get("repeat_run_char_ratio", 0.0))),
            "whitespace_balance": whitespace_quality,
        }

        raw_axes = [
            "alpha_ratio",
            "digit_ratio",
            "punct_ratio",
            "control_ratio",
            "whitespace_ratio",
            "repeat_run_char_ratio",
            "oov_ratio",
            "lm_log_ppl_scaled",
        ]
        raw_scaled_values = {
            "alpha_ratio": clamp01(float(raw.get("alpha_ratio", 0.0))),
            "digit_ratio": clamp01(float(raw.get("digit_ratio", 0.0))),
            "punct_ratio": clamp01(float(raw.get("punct_ratio", 0.0))),
            "control_ratio": clamp01(float(raw.get("control_ratio", 0.0))),
            "whitespace_ratio": clamp01(float(raw.get("whitespace_ratio", 0.0))),
            "repeat_run_char_ratio": clamp01(float(raw.get("repeat_run_char_ratio", 0.0))),
            "oov_ratio": clamp01(oov_ratio),
            "lm_log_ppl_scaled": lm_log_ppl_scaled,
        }

        # --- Option B: LM window-distribution spokes ---
        # These are useful for OCR where text quality can vary across a page/document:
        # - mean loss: overall language-likeness
        # - p10 loss: "best chunk" (does *any* good text exist?)
        # - p90 loss: "worst chunk" (are there catastrophic segments?)
        # - consistency: how stable the loss is across windows
        def loss_to_quality(loss_nats: Optional[float]) -> float:
            if loss_nats is None:
                return 0.0
            if isinstance(loss_nats, float) and not math.isfinite(loss_nats):
                return 0.0
            return clamp01((cfg.log_ppl_bad - float(loss_nats)) / (cfg.log_ppl_bad - cfg.log_ppl_good))

        window_count = int(lm.get("window_count", 0) or 0)
        lm_mean_quality = loss_to_quality(lm.get("window_loss_mean_nats"))
        lm_best_chunk_quality = loss_to_quality(lm.get("window_loss_p10_nats"))
        lm_worst_chunk_quality = loss_to_quality(lm.get("window_loss_p90_nats"))

        # Consistency: low std-dev of window losses => more consistent text
        win_std = lm.get("window_loss_std_nats", None)
        if window_count <= 1:
            lm_consistency = 1.0
        elif win_std is None or (isinstance(win_std, float) and not math.isfinite(win_std)):
            lm_consistency = 0.0
        elif cfg.lm_window_std_bad <= cfg.lm_window_std_good:
            lm_consistency = 0.0
        else:
            lm_consistency = clamp01(
                (cfg.lm_window_std_bad - float(win_std)) / (cfg.lm_window_std_bad - cfg.lm_window_std_good)
            )

        quality_windowed_axes = [
            "lm_mean_quality",
            "lm_best_chunk_quality",
            "lm_worst_chunk_quality",
            "lm_consistency",
            "vocab_quality",
            "oov_ratio",
            "alpha_ratio",
            "no_digits",
            "digit_ratio",
            "no_punct",
            "punct_ratio",
            "no_control",
            "no_repeats",
            "whitespace_balance",
            "whitespace_ratio",
        ]
        quality_windowed_values = {
            "lm_mean_quality": lm_mean_quality,
            "lm_best_chunk_quality": lm_best_chunk_quality,
            "lm_worst_chunk_quality": lm_worst_chunk_quality,
            "lm_consistency": lm_consistency,
            "vocab_quality": clamp01(1.0 - oov_ratio),
            "alpha_ratio": clamp01(float(raw.get("alpha_ratio", 0.0))),
            "no_digits": clamp01(1.0 - float(raw.get("digit_ratio", 0.0))),
            "no_punct": clamp01(1.0 - float(raw.get("punct_ratio", 0.0))),
            "no_control": clamp01(1.0 - float(raw.get("control_ratio", 0.0))),
            "no_repeats": clamp01(1.0 - float(raw.get("repeat_run_char_ratio", 0.0))),
            "whitespace_balance": whitespace_quality,
            # Raw ratios
            "oov_ratio": clamp01(oov_ratio),
            "digit_ratio": clamp01(float(raw.get("digit_ratio", 0.0))),
            "punct_ratio": clamp01(float(raw.get("punct_ratio", 0.0))),
            "whitespace_ratio": clamp01(float(raw.get("whitespace_ratio", 0.0))),
        }

        # Lexical profile radar (mostly direct ratios; not all are 'higher is better')
        lexical_axes = [
            "lm_quality",
            "vocab_quality",
            "oov_ratio",
            "type_token_ratio",
            "content_type_token_ratio",
            "content_repeat_token_ratio",
            "repeat_run_char_ratio",
            "whitespace_ratio",
        ]
        lexical_values = {
            "lm_quality": lm_quality,
            "vocab_quality": clamp01(1.0 - oov_ratio),
            "oov_ratio": clamp01(oov_ratio),
            "type_token_ratio": clamp01(float(raw.get("type_token_ratio", 0.0))),
            "content_type_token_ratio": clamp01(float(raw.get("content_type_token_ratio", 0.0))),
            "content_repeat_token_ratio": clamp01(float(raw.get("content_repeat_token_ratio", 0.0))),
            "repeat_run_char_ratio": clamp01(float(raw.get("repeat_run_char_ratio", 0.0))),
            "whitespace_ratio": clamp01(float(raw.get("whitespace_ratio", 0.0))),
        }


        return {
            "quality": {
                "axes": quality_axes,
                "values": quality_values,
                "notes": "All axes are 0..1 where higher means more language-like.",
            },
            "raw_scaled": {
                "axes": raw_axes,
                "values": raw_scaled_values,
                "notes": (
                    "Axes are 0..1. These are mostly raw composition ratios. "
                    "'lm_log_ppl_scaled' increases with perplexity (worse). "
                    "Interpret carefully."
                ),
            },
            "lexical": {
                "axes": lexical_axes,
                "values": lexical_values,
                "notes": (
                    "Lexical profile (0..1). Includes type-token ratios and OOV ratio. "
                    "Not all axes imply 'higher is better' (e.g., oov_ratio, content_repeat_token_ratio). "
                    "MinHash signature is stored under analysis.lexical.minhash for cross-item overlap estimation."
                ),
            },
            "quality_windowed": {
                "axes": quality_windowed_axes,
                "values": quality_windowed_values,
                "notes": (
                    "All axes are 0..1. Most axes are scaled so higher means more language-like. "
                    "The added raw ratio axes (oov_ratio, digit_ratio, punct_ratio, whitespace_ratio) are direct composition ratios "
                    "and are not necessarily 'higher is better'. "
                    "LM spokes use sliding-window loss distribution (mean/p10/p90) plus a consistency spoke (low std-dev)."
                ),
            },
        }


# ----------------------------
# JSON I/O
# ----------------------------

def write_json_report(report: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Path(report["file"]["basename"]).stem
    sha = report["file"]["sha256"][:12]
    out_path = output_dir / f"{base}__langlikeness__{sha}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path

def load_json_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Radar plotting
# ----------------------------

class RadarPlotter:
    def __init__(self) -> None:
        self.fig = Figure(figsize=(6.5, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111, polar=True)

    def clear(self) -> None:
        self.ax.clear()

    def plot_reports(
        self,
        reports: List[Dict[str, Any]],
        mode: str = "quality",
        overlay_alpha: float = 0.18,
        show_legend: bool = True,
    ) -> None:
        """
        mode: "quality" or "raw_scaled"
        """
        self.clear()

        if not reports:
            self.ax.set_title("No reports loaded")
            self.fig.tight_layout()
            return

        # Use axes from the first report; filter others to match
        first_radar = reports[0].get("analysis", {}).get("radar", {})
        if mode not in first_radar:
            self.ax.set_title(f"Missing radar mode: {mode}")
            self.fig.tight_layout()
            return

        axes = first_radar[mode]["axes"]
        N = len(axes)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        # style
        self.ax.set_theta_offset(np.pi / 2)
        self.ax.set_theta_direction(-1)
        self.ax.set_rlabel_position(0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.grid(True)

        labels = axes
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(labels, fontsize=9)

        # Plot each report
        for idx, rep in enumerate(reports):
            radar = rep.get("analysis", {}).get("radar", {})
            if mode not in radar:
                continue
            vals_map = radar[mode]["values"]
            vals = [float(vals_map.get(a, 0.0)) for a in axes]
            vals += vals[:1]

            name = rep.get("file", {}).get("basename", f"report-{idx+1}")
            alpha = overlay_alpha if len(reports) > 1 else 0.30
            self.ax.plot(angles, vals, linewidth=1.8, label=name)
            self.ax.fill(angles, vals, alpha=alpha)

        if mode == "quality":
            title = "Language-likeness radar (quality)"
        elif mode == "raw_scaled":
            title = "Language-likeness radar (raw/scaled)"
        elif mode == "quality_windowed":
            title = "Language-likeness radar (quality + LM windows + raw ratios)"
        elif mode == "lexical":
            title = "Language-likeness radar (lexical profile)"
        else:
            title = f"Language-likeness radar ({mode})"
        self.ax.set_title(title, pad=18)

        if show_legend:
            # If many reports, legend becomes huge; show only for <= 12
            if len(reports) <= 12:
                self.ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8, frameon=False)
            else:
                self.ax.text(
                    0.5, -0.15,
                    f"{len(reports)} overlays (legend hidden for readability)",
                    transform=self.ax.transAxes,
                    ha="center", va="center", fontsize=9,
                )

        self.fig.tight_layout()


# ----------------------------
# Tkinter GUI
# ----------------------------

class LangLikenessApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Language-Likeness Evaluator (OCR Transcript QA)")
        self.geometry("1150x720")
        self.minsize(980, 620)

        self.cfg = AnalyzerConfig()

        # state
        self.selected_text_files: List[Path] = []
        self.output_dir: Optional[Path] = None

        self.loaded_reports: List[Tuple[Path, Dict[str, Any]]] = []

        # batch visualization state (directory-loaded reports)
        self.batch_dir: Optional[Path] = None
        self.batch_reports: List[Tuple[Path, Dict[str, Any]]] = []
        self._batch_heatmap_col_to_report_idx: List[int] = []
        self._last_batch_composition_summary: Optional[Dict[str, Any]] = None

        # worker communication
        self._worker_thread: Optional[threading.Thread] = None
        self._q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

        self._build_ui()
        self.after(100, self._poll_worker_queue)

    def _build_ui(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.eval_frame = ttk.Frame(nb)
        self.viz_frame = ttk.Frame(nb)
        self.batch_frame = ttk.Frame(nb)

        nb.add(self.eval_frame, text="Evaluate Text Files")
        nb.add(self.viz_frame, text="Visualize JSON Reports")
        nb.add(self.batch_frame, text="Batch Visualizer")

        self._build_eval_tab(self.eval_frame)
        self._build_viz_tab(self.viz_frame)
        self._build_batch_tab(self.batch_frame)

    # -------------
    # Evaluate tab
    # -------------

    def _build_eval_tab(self, parent: ttk.Frame) -> None:
        # Layout: left controls, right log
        left = ttk.Frame(parent, padding=10)
        left.pack(side="left", fill="y")

        right = ttk.Frame(parent, padding=10)
        right.pack(side="right", fill="both", expand=True)

        # File selection
        ttk.Label(left, text="Input text files").pack(anchor="w")
        btn_files = ttk.Button(left, text="Select .txt files…", command=self._select_text_files)
        btn_files.pack(fill="x", pady=(4, 6))

        self.files_list = tk.Listbox(left, height=12, selectmode=tk.EXTENDED)
        self.files_list.pack(fill="x", pady=(0, 10))

        btn_clear = ttk.Button(left, text="Clear selection", command=self._clear_text_files)
        btn_clear.pack(fill="x", pady=(0, 14))

        # Output folder
        ttk.Label(left, text="Output folder for JSON reports").pack(anchor="w")
        out_row = ttk.Frame(left)
        out_row.pack(fill="x", pady=(4, 10))

        self.out_dir_var = tk.StringVar(value="")
        out_entry = ttk.Entry(out_row, textvariable=self.out_dir_var)
        out_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="Browse…", command=self._select_output_dir).pack(side="left", padx=(6, 0))

        # Options (LM + analysis)
        opts = ttk.LabelFrame(left, text="Options", padding=10)
        opts.pack(fill="x", pady=(0, 12))

        # Model
        ttk.Label(opts, text="LM model (Hugging Face):").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=self.cfg.model_name)
        ttk.Entry(opts, textvariable=self.model_var, width=28).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Device
        ttk.Label(opts, text="Device:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.device_var = tk.StringVar(value=self.cfg.device)
        ttk.Combobox(opts, textvariable=self.device_var, values=["auto", "cpu", "cuda"], state="readonly", width=10)\
            .grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        # Max chars
        ttk.Label(opts, text="Max chars to analyze:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.max_chars_var = tk.IntVar(value=self.cfg.max_chars)
        ttk.Spinbox(opts, from_=1000, to=10_000_000, increment=1000, textvariable=self.max_chars_var, width=12)\
            .grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        # Repeat run
        ttk.Label(opts, text="Min repeat run length:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.min_run_var = tk.IntVar(value=self.cfg.min_repeat_run)
        ttk.Spinbox(opts, from_=2, to=20, increment=1, textvariable=self.min_run_var, width=12)\
            .grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        self.exclude_ws_var = tk.BooleanVar(value=self.cfg.repeat_exclude_whitespace)
        ttk.Checkbutton(opts, text="Exclude whitespace runs from repeat metric", variable=self.exclude_ws_var)\
            .grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        opts.columnconfigure(1, weight=1)

        # Run
        self.run_btn = ttk.Button(left, text="Run evaluation → write JSON reports", command=self._run_evaluation)
        self.run_btn.pack(fill="x", pady=(0, 8))

        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 10))

        # Log
        ttk.Label(right, text="Log").pack(anchor="w")
        self.log = tk.Text(right, height=25, wrap="word")
        self.log.pack(fill="both", expand=True)
        self._log_line("Ready.")

    def _select_text_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select text files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not paths:
            return
        self.selected_text_files = [Path(p) for p in paths]
        self.files_list.delete(0, tk.END)
        for p in self.selected_text_files:
            self.files_list.insert(tk.END, str(p))

        self._log_line(f"Selected {len(self.selected_text_files)} file(s).")

    def _clear_text_files(self) -> None:
        self.selected_text_files = []
        self.files_list.delete(0, tk.END)
        self._log_line("Cleared file selection.")

    def _select_output_dir(self) -> None:
        p = filedialog.askdirectory(title="Select output folder for JSON reports")
        if not p:
            return
        self.output_dir = Path(p)
        self.out_dir_var.set(str(self.output_dir))
        self._log_line(f"Output folder set: {self.output_dir}")

    def _apply_cfg_from_ui(self) -> None:
        self.cfg.model_name = self.model_var.get().strip() or "distilgpt2"
        self.cfg.device = self.device_var.get()
        self.cfg.max_chars = int(self.max_chars_var.get())
        self.cfg.min_repeat_run = int(self.min_run_var.get())
        self.cfg.repeat_exclude_whitespace = bool(self.exclude_ws_var.get())

    def _run_evaluation(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Busy", "Evaluation is already running.")
            return

        if not self.selected_text_files:
            messagebox.showwarning("No files", "Select one or more text files first.")
            return

        if not self.out_dir_var.get().strip():
            messagebox.showwarning("No output folder", "Select an output folder for JSON reports.")
            return

        out_dir = Path(self.out_dir_var.get().strip())
        if not out_dir.exists():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Output folder error", f"Could not create output folder:\n{e}")
                return

        self.output_dir = out_dir
        self._apply_cfg_from_ui()

        # Pre-check optional dependencies so we fail fast on the UI thread
        try:
            _ = EnglishDictionary()
        except Exception as e:
            messagebox.showerror("Missing dependency", str(e))
            return
        try:
            # This will only import; model load happens in worker
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            messagebox.showerror(
                "Missing dependency",
                "Language model scoring requires 'transformers' and 'torch'.\n"
                "Install with: pip install transformers torch"
            )
            return

        self.run_btn.configure(state="disabled")
        self.progress.configure(value=0, maximum=len(self.selected_text_files))

        self._worker_thread = threading.Thread(
            target=self._worker_evaluate_files,
            args=(self.selected_text_files, self.output_dir, self.cfg),
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_evaluate_files(self, files: List[Path], out_dir: Path, cfg: AnalyzerConfig) -> None:
        try:
            analyzer = TextAnalyzer(cfg)
        except Exception as e:
            self._q.put(("error", f"Failed to initialize analyzer:\n{e}"))
            self._q.put(("done", None))
            return

        written: List[Path] = []
        for idx, p in enumerate(files, start=1):
            try:
                self._q.put(("log", f"[{idx}/{len(files)}] Analyzing: {p}"))
                rep = analyzer.analyze_file(p)
                out_path = write_json_report(rep, out_dir)
                written.append(out_path)
                self._q.put(("log", f"  → wrote {out_path.name}"))
            except Exception as e:
                self._q.put(("log", f"  ✗ error: {e}"))

            self._q.put(("progress", idx))

        self._q.put(("log", f"Done. Wrote {len(written)} report(s) to: {out_dir}"))
        self._q.put(("done", None))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self._log_line(str(payload))
                elif kind == "progress":
                    self.progress.configure(value=int(payload))
                elif kind == "error":
                    messagebox.showerror("Error", str(payload))
                elif kind == "done":
                    self.run_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.after(100, self._poll_worker_queue)

    def _log_line(self, s: str) -> None:
        self.log.insert(tk.END, s + "\n")
        self.log.see(tk.END)

    # -------------
    # Visualize tab
    # -------------

    def _build_viz_tab(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent, padding=10)
        left.pack(side="left", fill="y")

        right = ttk.Frame(parent, padding=10)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="JSON report files").pack(anchor="w")
        ttk.Button(left, text="Load JSON reports…", command=self._load_reports).pack(fill="x", pady=(4, 6))

        self.reports_list = tk.Listbox(left, height=14, selectmode=tk.EXTENDED)
        self.reports_list.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(0, 10))
        ttk.Button(row, text="Plot selected", command=self._plot_selected).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Plot all", command=self._plot_all).pack(side="left", fill="x", expand=True, padx=(6, 0))

        ttk.Button(left, text="Clear loaded reports", command=self._clear_reports).pack(fill="x", pady=(0, 12))

        # Mode
        mode_box = ttk.LabelFrame(left, text="Radar mode", padding=10)
        mode_box.pack(fill="x", pady=(0, 12))
        self.radar_mode_var = tk.StringVar(value="quality")
        ttk.Radiobutton(mode_box, text="Quality (higher=better)", value="quality", variable=self.radar_mode_var).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="Quality + LM windows (+ raw ratios)", value="quality_windowed", variable=self.radar_mode_var).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="Raw/scaled composition", value="raw_scaled", variable=self.radar_mode_var).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="Lexical profile", value="lexical", variable=self.radar_mode_var).pack(anchor="w")

        # Save plot
        ttk.Button(left, text="Save plot as PNG…", command=self._save_plot_png).pack(fill="x")

        # Plot area
        self.plotter = RadarPlotter()
        self.canvas = FigureCanvasTkAgg(self.plotter.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.plotter.plot_reports([], mode="quality")
        self.canvas.draw()

    def _load_reports(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Load JSON report files",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not paths:
            return

        added = 0
        for p in paths:
            path = Path(p)
            try:
                rep = load_json_report(path)
                self.loaded_reports.append((path, rep))
                added += 1
            except Exception as e:
                messagebox.showwarning("Load error", f"Failed to load {path.name}:\n{e}")

        if added:
            self._refresh_reports_list()
            messagebox.showinfo("Loaded", f"Loaded {added} report(s).")

    def _refresh_reports_list(self) -> None:
        self.reports_list.delete(0, tk.END)
        for path, rep in self.loaded_reports:
            base = rep.get("file", {}).get("basename", path.name)
            self.reports_list.insert(tk.END, f"{base}  [{path.name}]")

    def _clear_reports(self) -> None:
        self.loaded_reports = []
        self._refresh_reports_list()
        self.plotter.plot_reports([], mode=self.radar_mode_var.get())
        self.canvas.draw()

    def _plot_selected(self) -> None:
        sel = list(self.reports_list.curselection())
        if not sel:
            messagebox.showwarning("No selection", "Select one or more loaded reports first.")
            return
        reps = [self.loaded_reports[i][1] for i in sel]
        self.plotter.plot_reports(reps, mode=self.radar_mode_var.get())
        self.canvas.draw()

    def _plot_all(self) -> None:
        if not self.loaded_reports:
            messagebox.showwarning("No reports", "Load one or more report files first.")
            return
        reps = [r for _, r in self.loaded_reports]
        self.plotter.plot_reports(reps, mode=self.radar_mode_var.get())
        self.canvas.draw()

    def _save_plot_png(self) -> None:
        if not self.loaded_reports:
            messagebox.showwarning("No plot", "Load and plot one or more reports first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save radar plot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return
        try:
            self.plotter.fig.savefig(path, dpi=200, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Saved plot to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save PNG:\n{e}")

    # -------------
    # Batch visualizer tab
    # -------------

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        """
        Batch visualization for a directory of JSON report files:
        - Orders items by filename pattern (regex-based) or natural order fallback
        - Stacked tracks with optional rolling median
        - Feature heatmap with click-to-open radar popup for the selected item
        """
        left = ttk.Frame(parent, padding=10)
        left.pack(side="left", fill="y")

        right = ttk.Frame(parent, padding=10)
        right.pack(side="right", fill="both", expand=True)

        # Directory selection
        ttk.Label(left, text="Report directory (.json)").pack(anchor="w")
        dir_row = ttk.Frame(left)
        dir_row.pack(fill="x", pady=(4, 8))

        self.batch_dir_var = tk.StringVar(value="")
        ttk.Entry(dir_row, textvariable=self.batch_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(dir_row, text="Browse…", command=self._select_batch_dir).pack(side="left", padx=(6, 0))

        ttk.Button(left, text="Load directory", command=self._load_batch_dir_from_ui).pack(fill="x", pady=(0, 12))

        # Sorting options
        sort_box = ttk.LabelFrame(left, text="Ordering", padding=10)
        sort_box.pack(fill="x", pady=(0, 12))

        ttk.Label(sort_box, text="Sort regex (captures item number):").grid(row=0, column=0, sticky="w")
        # Default: last integer group in the name
        self.batch_sort_regex_var = tk.StringVar(value=r"(\d+)(?!.*\d)")
        ttk.Entry(sort_box, textvariable=self.batch_sort_regex_var).grid(row=1, column=0, sticky="ew", pady=(4, 0))

        ttk.Label(sort_box, text="Example: r\"(\\d+)(?!.*\\d)\"  (last digits)").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )

        sort_box.columnconfigure(0, weight=1)

        # Feature selection
        feat_box = ttk.LabelFrame(left, text="Features to plot", padding=10)
        feat_box.pack(fill="x", pady=(0, 12))

        self._batch_feature_keys = [
            "lm_mean_quality",
            "lm_best_chunk_quality",
            "lm_worst_chunk_quality",
            "lm_consistency",
            "vocab_quality",
            "oov_ratio",
            "type_token_ratio",
            "content_type_token_ratio",
            "content_repeat_token_ratio",
            "lexical_prev_jaccard_est",
            "lexical_prev_dice_est",
            "lexical_prev_overlap_coeff_est",
            "alpha_ratio",
            "digit_ratio",
            "punct_ratio",
            "whitespace_ratio",
            "whitespace_balance",
            "repeat_run_char_ratio",
            "control_ratio",
            "token_count_scaled",
            "analyzed_chars_scaled",
        ]
        self.batch_features_list = tk.Listbox(feat_box, height=12, selectmode=tk.EXTENDED)
        self.batch_features_list.pack(fill="x")
        for k in self._batch_feature_keys:
            self.batch_features_list.insert(tk.END, k)

        btn_row = ttk.Frame(feat_box)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="Select defaults", command=self._batch_select_default_features).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(btn_row, text="Select all", command=lambda: self.batch_features_list.select_set(0, tk.END)).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )

        # Smoothing options for tracks
        smooth_box = ttk.LabelFrame(left, text="Tracks options", padding=10)
        smooth_box.pack(fill="x", pady=(0, 12))

        self.batch_show_raw_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(smooth_box, text="Show per-item line", variable=self.batch_show_raw_var).grid(
            row=0, column=0, sticky="w"
        )

        self.batch_show_median_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(smooth_box, text="Show rolling median", variable=self.batch_show_median_var).grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )

        win_row = ttk.Frame(smooth_box)
        win_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(win_row, text="Median window:").pack(side="left")
        self.batch_median_window_var = tk.IntVar(value=11)
        ttk.Spinbox(win_row, from_=1, to=501, increment=2, textvariable=self.batch_median_window_var, width=8).pack(
            side="left", padx=(6, 0)
        )
        ttk.Label(win_row, text="(odd numbers recommended)").pack(side="left", padx=(8, 0))

        # Radar popup mode
        popup_box = ttk.LabelFrame(left, text="Click-to-radar popup", padding=10)
        popup_box.pack(fill="x", pady=(0, 12))

        self.batch_popup_radar_mode_var = tk.StringVar(value="quality_windowed")
        ttk.Label(popup_box, text="Radar mode:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            popup_box,
            textvariable=self.batch_popup_radar_mode_var,
            values=["quality_windowed", "quality", "raw_scaled", "lexical"],
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))
        popup_box.columnconfigure(1, weight=1)

        ttk.Button(left, text="Render batch plots", command=self._render_batch_plots).pack(fill="x")

        self.batch_status_var = tk.StringVar(value="No directory loaded.")
        ttk.Label(left, textvariable=self.batch_status_var, wraplength=260, justify="left").pack(anchor="w", pady=(10, 0))

        # Right: sub-notebook with Tracks, Heatmap, and Collection/Dataset Composition
        plot_nb = ttk.Notebook(right)
        plot_nb.pack(fill="both", expand=True)

        tracks_frame = ttk.Frame(plot_nb)
        heatmap_frame = ttk.Frame(plot_nb)
        composition_frame = ttk.Frame(plot_nb)

        plot_nb.add(tracks_frame, text="Stacked Tracks")
        plot_nb.add(heatmap_frame, text="Feature Heatmap")
        plot_nb.add(composition_frame, text="Dataset Composition")

        # Tracks figure/canvas
        self.batch_tracks_fig = Figure(figsize=(7.8, 6.4), dpi=100)
        self.batch_tracks_canvas = FigureCanvasTkAgg(self.batch_tracks_fig, master=tracks_frame)
        self.batch_tracks_canvas.draw()
        self.batch_tracks_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Heatmap figure/canvas
        self.batch_heatmap_fig = Figure(figsize=(7.8, 6.4), dpi=100)
        self.batch_heatmap_ax = self.batch_heatmap_fig.add_subplot(111)
        self.batch_heatmap_canvas = FigureCanvasTkAgg(self.batch_heatmap_fig, master=heatmap_frame)
        self.batch_heatmap_canvas.draw()
        self.batch_heatmap_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Dataset Composition tab: text summary + compact overview charts
        comp_controls = ttk.Frame(composition_frame, padding=(0, 0, 0, 6))
        comp_controls.pack(fill="x")
        ttk.Button(comp_controls, text="Refresh composition summary", command=self._render_batch_composition).pack(
            side="left"
        )
        ttk.Button(comp_controls, text="Save summary JSON…", command=self._save_batch_composition_json).pack(
            side="left", padx=(8, 0)
        )

        comp_text_frame = ttk.Frame(composition_frame)
        comp_text_frame.pack(fill="both", expand=False)
        self.batch_composition_text = tk.Text(comp_text_frame, height=16, wrap="word")
        comp_scroll = ttk.Scrollbar(comp_text_frame, orient="vertical", command=self.batch_composition_text.yview)
        self.batch_composition_text.configure(yscrollcommand=comp_scroll.set)
        self.batch_composition_text.pack(side="left", fill="both", expand=True)
        comp_scroll.pack(side="right", fill="y")
        self.batch_composition_text.insert(tk.END, "Load a report directory to generate collection-wide dataset composition metrics.\n")
        self.batch_composition_text.configure(state="disabled")

        self.batch_composition_fig = Figure(figsize=(7.8, 4.8), dpi=100)
        self.batch_composition_canvas = FigureCanvasTkAgg(self.batch_composition_fig, master=composition_frame)
        self.batch_composition_canvas.draw()
        self.batch_composition_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

        self._batch_heatmap_vline = None
        self.batch_heatmap_canvas.mpl_connect("button_press_event", self._on_batch_heatmap_click)

        # Default feature selection
        self._batch_select_default_features()

    def _batch_select_default_features(self) -> None:
        """Select a sensible default feature set for tracks/heatmap."""
        defaults = [
            "lm_mean_quality",
            "lm_best_chunk_quality",
            "lm_worst_chunk_quality",
            "lm_consistency",
            "oov_ratio",
            "lexical_prev_jaccard_est",
            "repeat_run_char_ratio",
            "whitespace_ratio",
            "digit_ratio",
            "punct_ratio",
            "token_count_scaled",
        ]
        self.batch_features_list.selection_clear(0, tk.END)
        for i, k in enumerate(self._batch_feature_keys):
            if k in defaults:
                self.batch_features_list.select_set(i)

    def _select_batch_dir(self) -> None:
        p = filedialog.askdirectory(title="Select directory containing JSON report files")
        if not p:
            return
        self.batch_dir_var.set(p)

    def _load_batch_dir_from_ui(self) -> None:
        s = self.batch_dir_var.get().strip()
        if not s:
            messagebox.showwarning("No directory", "Select a directory containing JSON report files first.")
            return
        d = Path(s)
        if not d.exists() or not d.is_dir():
            messagebox.showerror("Directory error", f"Not a valid directory:\n{d}")
            return
        self.batch_dir = d
        self._load_batch_reports_from_directory(d)

    def _load_batch_reports_from_directory(self, d: Path) -> None:
        json_paths = sorted(d.glob("*.json"), key=lambda p: natural_sort_key(p.name))
        reports: List[Tuple[Path, Dict[str, Any]]] = []
        bad = 0
        for p in json_paths:
            try:
                rep = load_json_report(p)
                reports.append((p, rep))
            except Exception:
                bad += 1

        self.batch_reports = reports
        self._batch_sort_reports()

        self.batch_status_var.set(
            f"Loaded {len(self.batch_reports)} report(s) from: {d}\n"
            f"{'(Some files failed to load.)' if bad else ''}"
        )

        # auto-render after load
        self._render_batch_plots()

    def _batch_sort_reports(self) -> None:
        pat = ""
        try:
            pat = self.batch_sort_regex_var.get().strip()
        except Exception:
            pat = ""

        def key_fn(t: Tuple[Path, Dict[str, Any]]) -> Any:
            p, rep = t
            name = rep.get("file", {}).get("basename") or rep.get("file", {}).get("path") or p.stem
            name = str(Path(name).name)
            return filename_pattern_sort_key(name, pat)

        self.batch_reports.sort(key=key_fn)
        self._batch_augment_reports_with_lexical_overlap()

    def _batch_augment_reports_with_lexical_overlap(self) -> None:
        """
        Computes adjacency lexical overlap estimates (Prev→Current) using stored MinHash signatures,
        and injects them into each report's analysis.features_raw so they can be plotted as tracks/heatmap.

        Derived per-item features (0..1, first item is NaN):
        - lexical_prev_jaccard_est
        - lexical_prev_dice_est
        - lexical_prev_overlap_coeff_est (estimated via Jaccard + unique-token counts)

        This does NOT write back to disk; it only augments the in-memory loaded reports.
        """
        reports = [rep for _, rep in getattr(self, "batch_reports", [])]
        N = len(reports)
        if N == 0:
            return

        sigs: List[Optional[List[str]]] = []
        uniq_sizes: List[int] = []

        for rep in reports:
            lex = rep.get("analysis", {}).get("lexical", {})
            # signature
            sig_hex = None
            try:
                sig_hex = lex.get("minhash", {}).get("signature_hex", None)
            except Exception:
                sig_hex = None

            if isinstance(sig_hex, list) and sig_hex:
                sigs.append([str(x) for x in sig_hex])
            else:
                sigs.append(None)

            # unique token counts (content tokens)
            try:
                ct = lex.get("content_tokens", {})
                uniq = int(ct.get("unique_token_count", 0) or 0)
            except Exception:
                uniq = 0
            uniq_sizes.append(max(0, uniq))

        for i, rep in enumerate(reports):
            fr = rep.setdefault("analysis", {}).setdefault("features_raw", {})

            if i == 0:
                fr["lexical_prev_jaccard_est"] = float("nan")
                fr["lexical_prev_dice_est"] = float("nan")
                fr["lexical_prev_overlap_coeff_est"] = float("nan")
                continue

            s_prev = sigs[i - 1]
            s_cur = sigs[i]
            if not s_prev or not s_cur or (len(s_prev) != len(s_cur)) or len(s_prev) == 0:
                fr["lexical_prev_jaccard_est"] = float("nan")
                fr["lexical_prev_dice_est"] = float("nan")
                fr["lexical_prev_overlap_coeff_est"] = float("nan")
                continue

            # MinHash Jaccard estimate
            matches = 0
            for a, b in zip(s_prev, s_cur):
                if a == b:
                    matches += 1
            jacc = matches / float(len(s_prev))
            if not math.isfinite(jacc):
                fr["lexical_prev_jaccard_est"] = float("nan")
                fr["lexical_prev_dice_est"] = float("nan")
                fr["lexical_prev_overlap_coeff_est"] = float("nan")
                continue

            jacc = float(np.clip(jacc, 0.0, 1.0))
            dice = float(np.clip((2.0 * jacc / (1.0 + jacc)) if (1.0 + jacc) > 0 else 0.0, 0.0, 1.0))

            # Estimate intersection size and overlap coefficient using unique token counts:
            # J = I / (a + b - I)  =>  I = J(a + b) / (1 + J)
            a = uniq_sizes[i - 1]
            b = uniq_sizes[i]
            if a > 0 and b > 0:
                inter_est = (jacc * float(a + b) / (1.0 + jacc)) if (1.0 + jacc) > 0 else 0.0
                overlap = inter_est / float(min(a, b)) if min(a, b) > 0 else float("nan")
                overlap = float(np.clip(overlap, 0.0, 1.0)) if math.isfinite(overlap) else float("nan")
            else:
                overlap = float("nan")

            fr["lexical_prev_jaccard_est"] = jacc
            fr["lexical_prev_dice_est"] = dice
            fr["lexical_prev_overlap_coeff_est"] = overlap

    def _batch_get_selected_feature_keys(self) -> List[str]:
        sel = list(self.batch_features_list.curselection())
        if not sel:
            # fallback to defaults if nothing selected
            self._batch_select_default_features()
            sel = list(self.batch_features_list.curselection())
        return [self._batch_feature_keys[i] for i in sel]

    def _batch_get_feature_value(self, report: Dict[str, Any], key: str) -> float:
        """
        Extracts a feature value in (mostly) 0..1 space for batch plotting.
        Missing values return NaN.
        """
        a = report.get("analysis", {})
        radar = a.get("radar", {})
        for mode in ("quality_windowed", "quality", "raw_scaled"):
            vals = radar.get(mode, {}).get("values", {})
            if key in vals:
                try:
                    return float(vals[key])
                except Exception:
                    return float("nan")

        raw = a.get("features_raw", {})
        if key in raw:
            try:
                v = float(raw[key])
                # features_raw are mostly ratios in 0..1; keep them as-is
                return v
            except Exception:
                return float("nan")

        if key == "token_count_scaled":
            tok = a.get("language_model", {}).get("token_count", None)
            try:
                tok = int(tok) if tok is not None else 0
            except Exception:
                tok = 0
            tokens_clip = 4096
            return clamp01(float(math.log1p(tok) / math.log1p(tokens_clip)))

        if key == "analyzed_chars_scaled":
            chars = a.get("analyzed_chars", None)
            try:
                chars = int(chars) if chars is not None else 0
            except Exception:
                chars = 0
            clip = max(1, int(self.cfg.max_chars))
            return clamp01(float(math.log1p(chars) / math.log1p(clip)))

        return float("nan")

    def _batch_numeric_array_from_reports(self, reports: List[Dict[str, Any]], getter: Any) -> np.ndarray:
        """Builds a float array from reports; missing/non-numeric values become NaN."""
        vals: List[float] = []
        for rep in reports:
            try:
                v = getter(rep)
                if v is None:
                    vals.append(float("nan"))
                else:
                    vals.append(float(v))
            except Exception:
                vals.append(float("nan"))
        return np.array(vals, dtype=float)

    def _batch_stats(self, values: Any) -> Dict[str, Any]:
        """Returns JSON-safe descriptive stats for a numeric sequence."""
        arr = np.asarray(values, dtype=float)
        total_n = int(arr.size)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {
                "n": 0,
                "missing": total_n,
                "sum": None,
                "min": None,
                "p10": None,
                "p25": None,
                "median": None,
                "mean": None,
                "p75": None,
                "p90": None,
                "max": None,
            }
        return {
            "n": int(finite.size),
            "missing": int(total_n - finite.size),
            "sum": float(np.sum(finite)),
            "min": float(np.min(finite)),
            "p10": float(np.percentile(finite, 10)),
            "p25": float(np.percentile(finite, 25)),
            "median": float(np.percentile(finite, 50)),
            "mean": float(np.mean(finite)),
            "p75": float(np.percentile(finite, 75)),
            "p90": float(np.percentile(finite, 90)),
            "max": float(np.max(finite)),
        }

    def _batch_fmt_num(self, value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            v = float(value)
        except Exception:
            return "n/a"
        if not math.isfinite(v):
            return "n/a"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:,.1f}"
        if abs(v) >= 10:
            return f"{v:,.2f}"
        return f"{v:,.{digits}f}"

    def _batch_fmt_pct(self, count: Any, total: int) -> str:
        try:
            c = int(count)
        except Exception:
            c = 0
        if total <= 0:
            return f"{c:,} (n/a)"
        return f"{c:,} ({100.0 * c / total:.1f}%)"

    def _batch_fmt_stats(self, stats: Dict[str, Any], digits: int = 3) -> str:
        if not stats or int(stats.get("n", 0) or 0) == 0:
            return "n=0"
        return (
            f"n={int(stats.get('n', 0)):,}, "
            f"median={self._batch_fmt_num(stats.get('median'), digits)}, "
            f"p10–p90={self._batch_fmt_num(stats.get('p10'), digits)}–{self._batch_fmt_num(stats.get('p90'), digits)}, "
            f"mean={self._batch_fmt_num(stats.get('mean'), digits)}, "
            f"min–max={self._batch_fmt_num(stats.get('min'), digits)}–{self._batch_fmt_num(stats.get('max'), digits)}"
        )

    def _batch_item_name(self, idx: int) -> str:
        try:
            p, rep = self.batch_reports[idx]
            return str(rep.get("file", {}).get("basename") or p.name)
        except Exception:
            return f"item-{idx + 1}"

    def _batch_longest_true_edge_run(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        For an adjacency mask where mask[i] means item i is similar to item i-1,
        returns the longest contiguous run of True edges as item indices.
        """
        best_edges = 0
        best_start = None
        best_end = None
        cur_edges = 0
        cur_start = None

        for i, flag in enumerate(mask):
            if i == 0:
                continue
            if bool(flag):
                if cur_start is None:
                    cur_start = i - 1
                    cur_edges = 1
                else:
                    cur_edges += 1
                cur_end = i
                if cur_edges > best_edges:
                    best_edges = cur_edges
                    best_start = cur_start
                    best_end = cur_end
            else:
                cur_start = None
                cur_edges = 0

        if best_edges <= 0 or best_start is None or best_end is None:
            return {
                "edge_count": 0,
                "item_count": 0,
                "start_index_1based": None,
                "end_index_1based": None,
                "start_item": None,
                "end_item": None,
            }

        return {
            "edge_count": int(best_edges),
            "item_count": int(best_edges + 1),
            "start_index_1based": int(best_start + 1),
            "end_index_1based": int(best_end + 1),
            "start_item": self._batch_item_name(best_start),
            "end_item": self._batch_item_name(best_end),
        }

    def _build_batch_composition_summary(self) -> Dict[str, Any]:
        """
        Builds collection-wide / dataset-level composition numbers from loaded JSON reports.
        The summary is intentionally JSON-safe so it can be exported and reused downstream.
        """
        reports = [rep for _, rep in self.batch_reports]
        N = len(reports)
        pat = self.batch_sort_regex_var.get().strip() if hasattr(self, "batch_sort_regex_var") else ""

        def arr(getter: Any) -> np.ndarray:
            return self._batch_numeric_array_from_reports(reports, getter)

        bytes_arr = arr(lambda r: r.get("file", {}).get("bytes", None))
        analyzed_chars = arr(lambda r: r.get("analysis", {}).get("analyzed_chars", None))
        dict_tokens = arr(lambda r: r.get("analysis", {}).get("dictionary", {}).get("token_count", None))
        lm_tokens = arr(lambda r: r.get("analysis", {}).get("language_model", {}).get("token_count", None))
        lex_tokens = arr(lambda r: r.get("analysis", {}).get("lexical", {}).get("tokens", {}).get("token_count", None))
        lex_unique = arr(lambda r: r.get("analysis", {}).get("lexical", {}).get("tokens", {}).get("unique_token_count", None))
        content_tokens = arr(lambda r: r.get("analysis", {}).get("lexical", {}).get("content_tokens", {}).get("token_count", None))
        content_unique = arr(lambda r: r.get("analysis", {}).get("lexical", {}).get("content_tokens", {}).get("unique_token_count", None))

        feature_keys = [
            "lm_mean_quality",
            "lm_best_chunk_quality",
            "lm_worst_chunk_quality",
            "lm_consistency",
            "vocab_quality",
            "oov_ratio",
            "type_token_ratio",
            "content_type_token_ratio",
            "content_repeat_token_ratio",
            "alpha_ratio",
            "digit_ratio",
            "punct_ratio",
            "whitespace_ratio",
            "whitespace_balance",
            "repeat_run_char_ratio",
            "control_ratio",
            "lexical_prev_jaccard_est",
            "lexical_prev_dice_est",
            "lexical_prev_overlap_coeff_est",
        ]
        feature_arrays = {
            k: arr(lambda r, key=k: self._batch_get_feature_value(r, key))
            for k in feature_keys
        }

        # Weighted character composition by total characters across the collection.
        char_fields = ["alpha", "digit", "punct", "control", "whitespace", "other", "non_whitespace"]
        char_totals = {k: 0 for k in char_fields}
        total_char_count = 0
        for rep in reports:
            cc = rep.get("analysis", {}).get("char_counts", {})
            try:
                total_char_count += int(cc.get("total", 0) or 0)
            except Exception:
                pass
            for k in char_fields:
                try:
                    char_totals[k] += int(cc.get(k, 0) or 0)
                except Exception:
                    pass
        weighted_char_ratios = {
            k: (float(v) / float(total_char_count) if total_char_count > 0 else None)
            for k, v in char_totals.items()
        }

        # Basic categorical heuristics for at-a-glance computational text utility.
        chars0 = np.nan_to_num(analyzed_chars, nan=0.0, posinf=0.0, neginf=0.0)
        word_tokens0 = np.nan_to_num(dict_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        q = feature_arrays["lm_mean_quality"]
        oov = feature_arrays["oov_ratio"]
        q_ok = np.isfinite(q)
        oov_ok = np.isfinite(oov)

        no_text = (chars0 <= 0) | (word_tokens0 <= 0)
        sparse = (~no_text) & ((word_tokens0 < 10) | (chars0 < 50))
        strong = (~no_text) & (~sparse) & q_ok & oov_ok & (word_tokens0 >= 20) & (q >= 0.70) & (oov <= 0.35)
        review = (~no_text) & (~sparse) & (~strong) & q_ok & oov_ok & (word_tokens0 >= 10) & (q >= 0.40) & (oov <= 0.60)
        low_noise = ~(no_text | sparse | strong | review)

        quality_counts = {
            "strong_computational_text": int(np.sum(strong)),
            "usable_or_review_text": int(np.sum(review)),
            "sparse_or_tiny_text": int(np.sum(sparse)),
            "empty_or_no_word_tokens": int(np.sum(no_text)),
            "low_likelihood_or_noise_like": int(np.sum(low_noise)),
        }

        truncated = 0
        schema_versions = set()
        model_names = set()
        created_values = []
        minhash_items = 0
        for rep in reports:
            try:
                if bool(rep.get("analysis", {}).get("truncated_to_max_chars", False)):
                    truncated += 1
            except Exception:
                pass
            schema_versions.add(str(rep.get("schema_version", "unknown")))
            lm = rep.get("analysis", {}).get("language_model", {})
            if lm.get("model_name"):
                model_names.add(str(lm.get("model_name")))
            if rep.get("created_at_utc"):
                created_values.append(str(rep.get("created_at_utc")))
            sig = rep.get("analysis", {}).get("lexical", {}).get("minhash", {}).get("signature_hex", None)
            if isinstance(sig, list) and len(sig) > 0:
                minhash_items += 1

        flag_counts = {
            "truncated_to_max_chars": int(truncated),
            "missing_lm_quality": int(np.sum(~q_ok)),
            "high_oov_ratio_gt_0_60": int(np.sum(np.isfinite(oov) & (oov > 0.60))),
            "high_repeat_run_ratio_gt_0_08": int(np.sum(np.isfinite(feature_arrays["repeat_run_char_ratio"]) & (feature_arrays["repeat_run_char_ratio"] > 0.08))),
            "control_ratio_gt_0_01": int(np.sum(np.isfinite(feature_arrays["control_ratio"]) & (feature_arrays["control_ratio"] > 0.01))),
            "digit_ratio_gt_0_30": int(np.sum(np.isfinite(feature_arrays["digit_ratio"]) & (feature_arrays["digit_ratio"] > 0.30))),
            "punct_ratio_gt_0_25": int(np.sum(np.isfinite(feature_arrays["punct_ratio"]) & (feature_arrays["punct_ratio"] > 0.25))),
            "whitespace_ratio_lt_0_05_or_gt_0_45": int(np.sum(np.isfinite(feature_arrays["whitespace_ratio"]) & ((feature_arrays["whitespace_ratio"] < 0.05) | (feature_arrays["whitespace_ratio"] > 0.45)))),
            "alpha_ratio_lt_0_50": int(np.sum(np.isfinite(feature_arrays["alpha_ratio"]) & (feature_arrays["alpha_ratio"] < 0.50))),
        }

        jacc = feature_arrays["lexical_prev_jaccard_est"]
        valid_jacc = np.isfinite(jacc)
        high_overlap_mask = valid_jacc & (jacc >= 0.75)
        medium_overlap_mask = valid_jacc & (jacc >= 0.50)
        low_overlap_mask = valid_jacc & (jacc <= 0.05)
        longest_high_overlap = self._batch_longest_true_edge_run(high_overlap_mask)

        top_pairs: List[Dict[str, Any]] = []
        for idx in np.where(valid_jacc)[0].tolist():
            if idx <= 0:
                continue
            top_pairs.append({
                "previous_index_1based": int(idx),
                "current_index_1based": int(idx + 1),
                "jaccard_est": float(jacc[idx]),
                "previous_item": self._batch_item_name(idx - 1),
                "current_item": self._batch_item_name(idx),
            })
        top_pairs.sort(key=lambda d: d.get("jaccard_est", 0.0), reverse=True)
        top_pairs = top_pairs[:8]

        first_item = self._batch_item_name(0) if N else None
        last_item = self._batch_item_name(N - 1) if N else None

        summary: Dict[str, Any] = {
            "summary_schema_version": 1,
            "created_at_utc": utc_now_iso(),
            "source_directory": str(self.batch_dir) if self.batch_dir else None,
            "item_count": int(N),
            "ordering": {
                "sort_regex": pat,
                "first_item": first_item,
                "last_item": last_item,
            },
            "report_metadata": {
                "schema_versions": sorted(schema_versions),
                "language_models": sorted(model_names),
                "earliest_report_created_at_utc": min(created_values) if created_values else None,
                "latest_report_created_at_utc": max(created_values) if created_values else None,
            },
            "coverage": {
                "total_source_bytes": int(np.nansum(bytes_arr)) if bytes_arr.size else 0,
                "total_analyzed_chars": int(np.nansum(analyzed_chars)) if analyzed_chars.size else 0,
                "total_word_tokens": int(np.nansum(dict_tokens)) if dict_tokens.size else 0,
                "total_lm_tokens": int(np.nansum(lm_tokens)) if lm_tokens.size else 0,
                "total_lexical_tokens": int(np.nansum(lex_tokens)) if lex_tokens.size else 0,
                "total_content_tokens": int(np.nansum(content_tokens)) if content_tokens.size else 0,
                "items_with_minhash_signature": int(minhash_items),
            },
            "length_distributions": {
                "source_bytes": self._batch_stats(bytes_arr),
                "analyzed_chars": self._batch_stats(analyzed_chars),
                "word_tokens": self._batch_stats(dict_tokens),
                "lm_tokens": self._batch_stats(lm_tokens),
                "lexical_tokens": self._batch_stats(lex_tokens),
                "content_tokens": self._batch_stats(content_tokens),
                "unique_lexical_tokens": self._batch_stats(lex_unique),
                "unique_content_tokens": self._batch_stats(content_unique),
            },
            "computational_text_utility": {
                "heuristic_class_counts": quality_counts,
                "heuristic_notes": {
                    "strong_computational_text": "word_tokens>=20, lm_mean_quality>=0.70, oov_ratio<=0.35",
                    "usable_or_review_text": "word_tokens>=10, lm_mean_quality>=0.40, oov_ratio<=0.60, excluding strong items",
                    "sparse_or_tiny_text": "some text but fewer than 10 word tokens or fewer than 50 analyzed characters",
                    "empty_or_no_word_tokens": "zero analyzed characters or zero dictionary word tokens",
                    "low_likelihood_or_noise_like": "remaining items not meeting the above text-utility heuristics",
                },
                "flag_counts": flag_counts,
            },
            "quality_and_composition_metrics": {
                k: self._batch_stats(v)
                for k, v in feature_arrays.items()
                if not k.startswith("lexical_prev_")
            },
            "character_composition": {
                "weighted_by_character_count": weighted_char_ratios,
                "raw_counts": {k: int(v) for k, v in char_totals.items()},
                "item_level_ratio_stats": {
                    k: self._batch_stats(feature_arrays[k])
                    for k in ["alpha_ratio", "digit_ratio", "punct_ratio", "whitespace_ratio", "control_ratio"]
                },
            },
            "lexical_profile": {
                "token_diversity_stats": {
                    "type_token_ratio": self._batch_stats(feature_arrays["type_token_ratio"]),
                    "content_type_token_ratio": self._batch_stats(feature_arrays["content_type_token_ratio"]),
                    "content_repeat_token_ratio": self._batch_stats(feature_arrays["content_repeat_token_ratio"]),
                },
                "adjacent_overlap": {
                    "valid_adjacent_comparisons": int(np.sum(valid_jacc)),
                    "jaccard_estimate": self._batch_stats(feature_arrays["lexical_prev_jaccard_est"]),
                    "dice_estimate": self._batch_stats(feature_arrays["lexical_prev_dice_est"]),
                    "overlap_coefficient_estimate": self._batch_stats(feature_arrays["lexical_prev_overlap_coeff_est"]),
                    "adjacent_pairs_jaccard_ge_0_75": int(np.sum(high_overlap_mask)),
                    "adjacent_pairs_jaccard_ge_0_50": int(np.sum(medium_overlap_mask)),
                    "adjacent_pairs_jaccard_le_0_05": int(np.sum(low_overlap_mask)),
                    "longest_jaccard_ge_0_75_run": longest_high_overlap,
                    "top_adjacent_overlap_pairs": top_pairs,
                },
            },
        }
        return summary

    def _format_batch_composition_summary(self, summary: Dict[str, Any]) -> str:
        N = int(summary.get("item_count", 0) or 0)
        lines: List[str] = []
        lines.append("COLLECTION / DATASET COMPOSITION SUMMARY")
        lines.append("=" * 52)
        lines.append(f"Source directory: {summary.get('source_directory') or 'n/a'}")
        lines.append(f"Reports / items loaded: {N:,}")
        ordering = summary.get("ordering", {})
        lines.append(f"Ordering regex: {ordering.get('sort_regex') or 'natural sort fallback'}")
        lines.append(f"First item: {ordering.get('first_item') or 'n/a'}")
        lines.append(f"Last item: {ordering.get('last_item') or 'n/a'}")

        meta = summary.get("report_metadata", {})
        lines.append("")
        lines.append("REPORT / ANALYSIS METADATA")
        lines.append("-" * 52)
        lines.append(f"Report schema versions: {', '.join(meta.get('schema_versions', [])) or 'n/a'}")
        lines.append(f"Language model(s): {', '.join(meta.get('language_models', [])) or 'n/a'}")
        lines.append(f"Earliest report timestamp: {meta.get('earliest_report_created_at_utc') or 'n/a'}")
        lines.append(f"Latest report timestamp: {meta.get('latest_report_created_at_utc') or 'n/a'}")

        cov = summary.get("coverage", {})
        lines.append("")
        lines.append("DATASET SCALE / COVERAGE")
        lines.append("-" * 52)
        lines.append(f"Total source bytes: {int(cov.get('total_source_bytes', 0) or 0):,}")
        lines.append(f"Total analyzed characters: {int(cov.get('total_analyzed_chars', 0) or 0):,}")
        lines.append(f"Total dictionary word tokens: {int(cov.get('total_word_tokens', 0) or 0):,}")
        lines.append(f"Total LM tokens: {int(cov.get('total_lm_tokens', 0) or 0):,}")
        lines.append(f"Total lexical content tokens: {int(cov.get('total_content_tokens', 0) or 0):,}")
        lines.append(f"Items with MinHash lexical signatures: {self._batch_fmt_pct(cov.get('items_with_minhash_signature', 0), N)}")

        lengths = summary.get("length_distributions", {})
        lines.append("")
        lines.append("LENGTH DISTRIBUTIONS")
        lines.append("-" * 52)
        for label, key in [
            ("Source bytes / item", "source_bytes"),
            ("Analyzed chars / item", "analyzed_chars"),
            ("Word tokens / item", "word_tokens"),
            ("LM tokens / item", "lm_tokens"),
            ("Content tokens / item", "content_tokens"),
            ("Unique content tokens / item", "unique_content_tokens"),
        ]:
            lines.append(f"{label}: {self._batch_fmt_stats(lengths.get(key, {}), digits=2)}")

        utility = summary.get("computational_text_utility", {})
        class_counts = utility.get("heuristic_class_counts", {})
        lines.append("")
        lines.append("COMPUTATIONAL TEXT UTILITY (HEURISTIC)")
        lines.append("-" * 52)
        for label, key in [
            ("Strong computational text", "strong_computational_text"),
            ("Usable / review text", "usable_or_review_text"),
            ("Sparse or tiny text", "sparse_or_tiny_text"),
            ("Empty / no word tokens", "empty_or_no_word_tokens"),
            ("Low-likelihood or noise-like", "low_likelihood_or_noise_like"),
        ]:
            lines.append(f"{label}: {self._batch_fmt_pct(class_counts.get(key, 0), N)}")

        flags = utility.get("flag_counts", {})
        lines.append("")
        lines.append("QUALITY / OCR WARNING FLAGS")
        lines.append("-" * 52)
        for label, key in [
            ("Truncated to max_chars", "truncated_to_max_chars"),
            ("Missing LM quality", "missing_lm_quality"),
            ("High OOV ratio (>0.60)", "high_oov_ratio_gt_0_60"),
            ("High repeated-run ratio (>0.08)", "high_repeat_run_ratio_gt_0_08"),
            ("Control characters (>1%)", "control_ratio_gt_0_01"),
            ("Digit-heavy (>30%)", "digit_ratio_gt_0_30"),
            ("Punctuation-heavy (>25%)", "punct_ratio_gt_0_25"),
            ("Extreme whitespace (<5% or >45%)", "whitespace_ratio_lt_0_05_or_gt_0_45"),
            ("Low alphabetic ratio (<50%)", "alpha_ratio_lt_0_50"),
        ]:
            lines.append(f"{label}: {self._batch_fmt_pct(flags.get(key, 0), N)}")

        qm = summary.get("quality_and_composition_metrics", {})
        lines.append("")
        lines.append("CORE FEATURE DISTRIBUTIONS")
        lines.append("-" * 52)
        for key in [
            "lm_mean_quality",
            "lm_best_chunk_quality",
            "lm_worst_chunk_quality",
            "lm_consistency",
            "oov_ratio",
            "repeat_run_char_ratio",
            "whitespace_ratio",
            "digit_ratio",
            "punct_ratio",
            "type_token_ratio",
            "content_type_token_ratio",
            "content_repeat_token_ratio",
        ]:
            if key in qm:
                lines.append(f"{key}: {self._batch_fmt_stats(qm.get(key, {}), digits=3)}")

        comp = summary.get("character_composition", {})
        weighted = comp.get("weighted_by_character_count", {})
        lines.append("")
        lines.append("CHARACTER COMPOSITION (WEIGHTED BY ALL CHARACTERS)")
        lines.append("-" * 52)
        for key in ["alpha", "digit", "punct", "whitespace", "control", "other", "non_whitespace"]:
            v = weighted.get(key)
            pct = "n/a" if v is None else f"{100.0 * float(v):.2f}%"
            lines.append(f"{key}: {pct}")

        lexical = summary.get("lexical_profile", {})
        overlap = lexical.get("adjacent_overlap", {})
        lines.append("")
        lines.append("LEXICAL PROFILE AND ADJACENT OVERLAP")
        lines.append("-" * 52)
        lines.append(f"Valid adjacent lexical comparisons: {int(overlap.get('valid_adjacent_comparisons', 0) or 0):,} of {max(0, N - 1):,}")
        lines.append(f"Adjacent Jaccard estimate: {self._batch_fmt_stats(overlap.get('jaccard_estimate', {}), digits=3)}")
        lines.append(f"Adjacent Dice estimate: {self._batch_fmt_stats(overlap.get('dice_estimate', {}), digits=3)}")
        lines.append(f"Adjacent overlap coefficient estimate: {self._batch_fmt_stats(overlap.get('overlap_coefficient_estimate', {}), digits=3)}")
        lines.append(f"Adjacent pairs with Jaccard ≥ 0.75: {self._batch_fmt_pct(overlap.get('adjacent_pairs_jaccard_ge_0_75', 0), max(1, N - 1))}")
        lines.append(f"Adjacent pairs with Jaccard ≥ 0.50: {self._batch_fmt_pct(overlap.get('adjacent_pairs_jaccard_ge_0_50', 0), max(1, N - 1))}")
        lines.append(f"Adjacent pairs with Jaccard ≤ 0.05: {self._batch_fmt_pct(overlap.get('adjacent_pairs_jaccard_le_0_05', 0), max(1, N - 1))}")

        run = overlap.get("longest_jaccard_ge_0_75_run", {})
        if int(run.get("item_count", 0) or 0) > 0:
            lines.append(
                "Longest high-overlap run (Jaccard ≥ 0.75): "
                f"{int(run.get('item_count', 0)):,} items, "
                f"#{run.get('start_index_1based')}–#{run.get('end_index_1based')} "
                f"({run.get('start_item')} → {run.get('end_item')})"
            )
        else:
            lines.append("Longest high-overlap run (Jaccard ≥ 0.75): none detected")

        top_pairs = overlap.get("top_adjacent_overlap_pairs", [])
        if top_pairs:
            lines.append("")
            lines.append("Top adjacent lexical-overlap pairs:")
            for pair in top_pairs[:5]:
                lines.append(
                    f"  J={pair.get('jaccard_est', 0.0):.3f}  "
                    f"#{pair.get('previous_index_1based')} {pair.get('previous_item')}  →  "
                    f"#{pair.get('current_index_1based')} {pair.get('current_item')}"
                )

        lines.append("")
        lines.append("INTERPRETATION NOTES")
        lines.append("-" * 52)
        lines.append("• Strong/usable/noise classes are heuristics for triage, not ground-truth OCR accuracy labels.")
        lines.append("• Length distributions are essential context because very short texts can produce unstable language-model and lexical scores.")
        lines.append("• Adjacent lexical overlap is computed after filename-pattern sorting, so it is meaningful only if the order reflects collection order.")
        lines.append("• High overlap can indicate duplicate text, forms/templates, multi-page continuity, repeated boilerplate, or repeated OCR artifacts.")
        return "\n".join(lines)

    def _json_safe(self, obj: Any) -> Any:
        """Recursively converts numpy scalars and NaN/Inf values into JSON-safe Python values."""
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            obj = float(obj)
        if isinstance(obj, float):
            if not math.isfinite(obj):
                return None
            return obj
        return obj

    def _render_batch_composition(self) -> None:
        if not getattr(self, "batch_reports", None):
            try:
                self.batch_composition_text.configure(state="normal")
                self.batch_composition_text.delete("1.0", tk.END)
                self.batch_composition_text.insert(tk.END, "Load a report directory to generate collection-wide dataset composition metrics.\n")
                self.batch_composition_text.configure(state="disabled")
            except Exception:
                pass
            try:
                self.batch_composition_fig.clear()
                ax = self.batch_composition_fig.add_subplot(111)
                ax.set_title("No reports loaded")
                self.batch_composition_canvas.draw()
            except Exception:
                pass
            return

        # Keep the summary aligned with the current collection-order regex.
        self._batch_sort_reports()
        summary = self._build_batch_composition_summary()
        self._last_batch_composition_summary = summary

        text = self._format_batch_composition_summary(summary)
        self.batch_composition_text.configure(state="normal")
        self.batch_composition_text.delete("1.0", tk.END)
        self.batch_composition_text.insert(tk.END, text)
        self.batch_composition_text.configure(state="disabled")

        self._render_batch_composition_charts(summary)

    def _render_batch_composition_charts(self, summary: Dict[str, Any]) -> None:
        reports = [rep for _, rep in self.batch_reports]
        N = len(reports)
        self.batch_composition_fig.clear()
        if N == 0:
            ax = self.batch_composition_fig.add_subplot(111)
            ax.set_title("No reports loaded")
            self.batch_composition_canvas.draw()
            return

        gs = self.batch_composition_fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35)

        # 1) Heuristic text utility categories
        ax1 = self.batch_composition_fig.add_subplot(gs[0, 0])
        class_counts = summary.get("computational_text_utility", {}).get("heuristic_class_counts", {})
        class_labels = ["strong", "review", "sparse", "empty", "low/noise"]
        class_keys = [
            "strong_computational_text",
            "usable_or_review_text",
            "sparse_or_tiny_text",
            "empty_or_no_word_tokens",
            "low_likelihood_or_noise_like",
        ]
        class_vals = [int(class_counts.get(k, 0) or 0) for k in class_keys]
        ax1.bar(np.arange(len(class_vals)), class_vals)
        ax1.set_xticks(np.arange(len(class_vals)))
        ax1.set_xticklabels(class_labels, rotation=25, ha="right", fontsize=8)
        ax1.set_ylabel("items")
        ax1.set_title("Computational text utility")

        # 2) Token-count distribution on log scale
        ax2 = self.batch_composition_fig.add_subplot(gs[0, 1])
        token_arr = self._batch_numeric_array_from_reports(
            reports, lambda r: r.get("analysis", {}).get("dictionary", {}).get("token_count", None)
        )
        finite_tokens = token_arr[np.isfinite(token_arr)]
        if finite_tokens.size:
            ax2.hist(np.log10(np.maximum(finite_tokens, 0.0) + 1.0), bins=min(30, max(5, int(math.sqrt(finite_tokens.size)))))
            ax2.set_xlabel("log10(word tokens + 1)")
            ax2.set_ylabel("items")
        else:
            ax2.text(0.5, 0.5, "No token data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Text length distribution")

        # 3) Weighted character composition
        ax3 = self.batch_composition_fig.add_subplot(gs[1, 0])
        weighted = summary.get("character_composition", {}).get("weighted_by_character_count", {})
        comp_keys = ["alpha", "digit", "punct", "whitespace", "control", "other"]
        comp_vals = [float(weighted.get(k) or 0.0) for k in comp_keys]
        ax3.bar(np.arange(len(comp_vals)), comp_vals)
        ax3.set_xticks(np.arange(len(comp_vals)))
        ax3.set_xticklabels(comp_keys, rotation=25, ha="right", fontsize=8)
        ax3.set_ylim(0.0, max(1.0, max(comp_vals) * 1.1 if comp_vals else 1.0))
        ax3.set_ylabel("share of characters")
        ax3.set_title("Character composition")

        # 4) Adjacent lexical overlap distribution
        ax4 = self.batch_composition_fig.add_subplot(gs[1, 1])
        jacc = np.array([self._batch_get_feature_value(rep, "lexical_prev_jaccard_est") for rep in reports], dtype=float)
        finite_jacc = jacc[np.isfinite(jacc)]
        if finite_jacc.size:
            ax4.hist(np.clip(finite_jacc, 0.0, 1.0), bins=np.linspace(0.0, 1.0, 21))
            ax4.set_xlabel("adjacent Jaccard estimate")
            ax4.set_ylabel("adjacent pairs")
        else:
            ax4.text(0.5, 0.5, "No adjacent overlap data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Lexical overlap in collection order")

        self.batch_composition_fig.suptitle(f"Dataset composition overview (N={N:,})", y=0.995)
        self.batch_composition_fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.batch_composition_canvas.draw()

    def _save_batch_composition_json(self) -> None:
        if not getattr(self, "batch_reports", None):
            messagebox.showwarning("No batch data", "Load a directory of JSON reports first.")
            return
        if self._last_batch_composition_summary is None:
            self._render_batch_composition()
        if self._last_batch_composition_summary is None:
            messagebox.showwarning("No summary", "No dataset composition summary is available yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save dataset composition summary",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._json_safe(self._last_batch_composition_summary), f, indent=2, ensure_ascii=False, allow_nan=False)
            messagebox.showinfo("Saved", f"Saved dataset composition summary to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save summary JSON:\n{e}")

    def _render_batch_plots(self) -> None:
        if not getattr(self, "batch_reports", None):
            messagebox.showwarning("No batch data", "Load a directory of JSON reports first.")
            return

        # Re-sort in case the pattern changed
        self._batch_sort_reports()

        # Render plots
        self._render_batch_tracks()
        self._render_batch_heatmap()
        self._render_batch_composition()

    def _render_batch_tracks(self) -> None:
        reports = [rep for _, rep in self.batch_reports]
        N = len(reports)
        if N == 0:
            self.batch_tracks_fig.clear()
            ax = self.batch_tracks_fig.add_subplot(111)
            ax.set_title("No reports loaded")
            self.batch_tracks_canvas.draw()
            return

        keys = self._batch_get_selected_feature_keys()

        show_raw = bool(self.batch_show_raw_var.get())
        show_med = bool(self.batch_show_median_var.get())
        win = int(self.batch_median_window_var.get() or 1)
        if win < 1:
            win = 1

        x = np.arange(N, dtype=float)

        self.batch_tracks_fig.clear()
        gs = self.batch_tracks_fig.add_gridspec(len(keys), 1, hspace=0.12)
        axes = []

        for i, k in enumerate(keys):
            ax = self.batch_tracks_fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
            axes.append(ax)

            y = np.array([self._batch_get_feature_value(rep, k) for rep in reports], dtype=float)

            # Clamp obvious ratio features into 0..1 for plotting (leave NaNs intact)
            y = np.where(np.isfinite(y), np.clip(y, 0.0, 1.0), np.nan)

            if show_raw:
                line = ax.plot(x, y, linewidth=0.8, alpha=0.35)[0]
            else:
                # create a dummy line to grab a consistent color
                line = ax.plot([], [], linewidth=0.8)[0]

            if show_med and win > 1:
                # Fill NaNs for smoothing (use median of finite values, else 0)
                finite = y[np.isfinite(y)]
                fill = float(np.median(finite)) if finite.size else 0.0
                yfill = np.where(np.isfinite(y), y, fill)
                med = rolling_median_1d(yfill, win)
                ax.plot(x, med, linewidth=1.6, alpha=0.90, color=line.get_color())

            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.set_ylabel(k, fontsize=8, rotation=0, labelpad=36, ha="right", va="center")

            if i < len(keys) - 1:
                ax.tick_params(labelbottom=False)

        axes[-1].set_xlabel("Item order (sorted by filename pattern)")
        title = f"Batch tracks (N={N})"
        self.batch_tracks_fig.suptitle(title, y=0.995)
        self.batch_tracks_fig.tight_layout(rect=[0, 0, 1, 0.985])
        self.batch_tracks_canvas.draw()

    def _render_batch_heatmap(self) -> None:
        reports = [rep for _, rep in self.batch_reports]
        N = len(reports)
        if N == 0:
            self.batch_heatmap_fig.clear()
            ax = self.batch_heatmap_fig.add_subplot(111)
            ax.set_title("No reports loaded")
            self.batch_heatmap_canvas.draw()
            return

        keys = self._batch_get_selected_feature_keys()

        # Build feature matrix (F x N)
        F = len(keys)
        mat = np.zeros((F, N), dtype=float)
        for i, k in enumerate(keys):
            row = np.array([self._batch_get_feature_value(rep, k) for rep in reports], dtype=float)
            row = np.where(np.isfinite(row), np.clip(row, 0.0, 1.0), 0.0)
            mat[i, :] = row

        # Auto-bin for very large N to keep the heatmap responsive.
        # When binned, each displayed column maps to a representative item index for click-to-radar.
        max_cols = 2500
        if N > max_cols:
            bin_size = int(math.ceil(N / max_cols))
            cols = []
            col_to_idx: List[int] = []
            for start in range(0, N, bin_size):
                end = min(N, start + bin_size)
                cols.append(np.median(mat[:, start:end], axis=1))
                col_to_idx.append((start + end - 1) // 2)  # representative item
            mat_disp = np.stack(cols, axis=1) if cols else mat[:, :1]
            self._batch_heatmap_col_to_report_idx = col_to_idx
            x_label = f"Item order (binned, bin_size={bin_size})"
        else:
            mat_disp = mat
            self._batch_heatmap_col_to_report_idx = list(range(N))
            x_label = "Item order (sorted)"

        self.batch_heatmap_fig.clear()
        ax = self.batch_heatmap_fig.add_subplot(111)
        self.batch_heatmap_ax = ax  # update reference for click handler

        im = ax.imshow(mat_disp, aspect="auto", interpolation="nearest", origin="lower")
        ax.set_yticks(np.arange(F))
        ax.set_yticklabels(keys, fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_title(f"Feature heatmap (N={N}) — click a column to open radar")

        # Colorbar (recreated each time for simplicity)
        cbar = self.batch_heatmap_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Value (0..1)", rotation=270, labelpad=14)

        self.batch_heatmap_fig.tight_layout()
        self.batch_heatmap_canvas.draw()

    def _on_batch_heatmap_click(self, event: Any) -> None:
        # Only handle clicks on the heatmap axis
        if event.inaxes is None or event.inaxes != getattr(self, "batch_heatmap_ax", None):
            return
        if event.xdata is None:
            return

        col = int(round(float(event.xdata)))
        if col < 0 or col >= len(self._batch_heatmap_col_to_report_idx):
            return

        idx = self._batch_heatmap_col_to_report_idx[col]
        if idx < 0 or idx >= len(self.batch_reports):
            return

        # Visual cue: vertical line at the selected column
        try:
            if self._batch_heatmap_vline is not None:
                self._batch_heatmap_vline.remove()
        except Exception:
            pass

        try:
            self._batch_heatmap_vline = self.batch_heatmap_ax.axvline(col, linewidth=1.0, alpha=0.8)
            self.batch_heatmap_canvas.draw_idle()
        except Exception:
            pass

        _, rep = self.batch_reports[idx]
        basename = rep.get("file", {}).get("basename", f"item-{idx+1}")
        self.batch_status_var.set(f"Selected: #{idx+1}  {basename}")

        mode = self.batch_popup_radar_mode_var.get() or "quality_windowed"
        self._open_radar_popup(rep, mode=mode, title=f"{basename} (#{idx+1})")

    def _open_radar_popup(self, report: Dict[str, Any], mode: str = "quality_windowed", title: str = "Radar") -> None:
        """
        Opens a new window with a radar plot for a single report.
        Falls back to 'quality' if the requested mode doesn't exist.
        """
        win = tk.Toplevel(self)
        win.title(f"Radar — {title}")
        win.geometry("720x680")

        # choose best available mode
        radar = report.get("analysis", {}).get("radar", {})
        if mode not in radar:
            mode = "quality" if "quality" in radar else ("raw_scaled" if "raw_scaled" in radar else mode)

        plotter = RadarPlotter()
        plotter.plot_reports([report], mode=mode, overlay_alpha=0.30, show_legend=False)

        canvas = FigureCanvasTkAgg(plotter.fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # prevent garbage collection
        win._radar_plotter = plotter  # type: ignore[attr-defined]
        win._radar_canvas = canvas    # type: ignore[attr-defined]


def main() -> None:
    # Ensure ttk styling looks decent cross-platform
    try:
        import ctypes  # noqa
    except Exception:
        pass
    app = LangLikenessApp()
    app.mainloop()


if __name__ == "__main__":
    main()
