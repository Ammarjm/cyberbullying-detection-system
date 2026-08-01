# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# ── Ensure UTF-8 output on Windows terminals (fixes UnicodeEncodeError for █ etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Project paths ─────────────────────────────────────────────────────────────
from pathlib import Path as _Path
PROJECT_DIR = _Path(__file__).resolve().parent
BASE_DIR = _Path(os.getenv("CB_BASE_DIR", PROJECT_DIR))
DATA_DIR = _Path(os.getenv("CB_DATA_DIR", PROJECT_DIR))
MODEL_DIR = _Path(os.getenv("CB_MODEL_DIR", BASE_DIR / "cyberbullying_bert_model"))



# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: SENSING LAYER  —  Data Structures & Input Definitions
# ─────────────────────────────────────────────────────────────────────────────

class CyberbullyingType(Enum):
    """Cyberbullying categories present in the Kaggle dataset."""
    AGE              = "age"
    ETHNICITY        = "ethnicity"
    GENDER           = "gender"
    RELIGION         = "religion"
    NOT_CYBERBULLYING = "not_cyberbullying"
    OTHER_CYBERBULLYING = "other_cyberbullying"


class ThreatLevel(Enum):
    """
    Threat levels produced by the Threshold-Based Decision Algorithm (§3.5.5).

    Decision formula:
        S = w1 * p_toxic + w2 * r_rule − w3 * C_context

    Thresholds (τ):
        CRITICAL  if  S ≥ τ_c
        HIGH      if  τ_h ≤ S < τ_c
        LOW       if  τ_l ≤ S < τ_h
        NOT_CB    if  S < τ_l
    """
    NOT_CB   = "NOT_CB"     # Not cyberbullying
    LOW      = "LOW"        # Low threat
    HIGH     = "HIGH"       # High threat
    CRITICAL = "CRITICAL"   # Critical threat


@dataclass
class DetectionResult:
    """Full analysis result for a single text input."""
    text:                 str
    p_toxic:              float   # BERT toxicity probability  (0–1)
    r_rule:               int     # Rule-based explicit flag     (0 or 1)
    r_rule_soft:          float   # Rule-based soft score        (0–1, includes indirect bias)
    c_context:            float   # Context mitigation score   (0–1)
    mitigation_triggered: bool    # Constructive criticism flag
    severity_score:       float   # Fused severity  S
    threat_level:         ThreatLevel
    predicted_type:       Optional[str] = None   # Fine-grained BERT class
    actions:              List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: PROCESSING LAYER  —  Text Preprocessing  (§3.4)
# ─────────────────────────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Implements all preprocessing steps described in §3.4:
      - Data Cleaning          (§3.4.1)
      - Tokenization ready for BERT  (§3.4.2)
    """

    def __init__(self):
        # Unicode normalisation map for common obfuscations
        self._leet_map = str.maketrans(
            "30|@1!",
            "eolaiI"
        )

    # ── public entry point ──────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        """Full cleaning pipeline → returns normalised string."""
        text = self._remove_urls(text)
        text = self._remove_non_semantic_symbols(text)
        text = self._reduce_character_repetition(text)
        text = self._normalise(text)
        return text.strip()

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove URLs and hyperlinks."""
        return re.sub(r"http\S+|www\.\S+", "", text)

    @staticmethod
    def _remove_non_semantic_symbols(text: str) -> str:
        """Remove non-semantic noise while preserving punctuation meaning."""
        text = re.sub(r"@\w+", "", text)            # mentions
        text = re.sub(r"#(\w+)", r"\1", text)       # keep hashtag text
        text = re.sub(r"[^\w\s.,!?'\"()-]", " ", text)
        return text

    @staticmethod
    def _reduce_character_repetition(text: str) -> str:
        """Reduce stretched words like 'sooooo' → 'soo'."""
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    def _normalise(self, text: str) -> str:
        """Lowercase + basic leet-speak normalisation."""
        text = text.lower()
        text = text.translate(self._leet_map)
        text = re.sub(r"\s+", " ", text)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: INTELLIGENCE LAYER
# ─────────────────────────────────────────────────────────────────────────────

# ── 3A. Core BERT Model  (§3.5.1) ───────────────────────────────────────────

class BERTToxicityAnalyzer:
    """
    Wraps a fine-tuned BERT classifier for multi-class cyberbullying detection.

    Architecture:
      - Base: bert-base-uncased  (Devlin et al., 2019)
      - Tokeniser: WordPiece  (§3.4.2)
      - Head: Linear(768 → num_classes)
      - Fine-tuned on the Kaggle Cyberbullying Classification Dataset

    Supports two HuggingFace models automatically:
      1. "unitary/toxic-bert"   – multi-label binary (toxic/severe_toxic/obscene/threat/insult/identity_hate)
      2. Multi-class models     – direct class probabilities

    When `transformers` is unavailable, falls back to a TF-Heuristic so the
    pipeline remains fully functional without a GPU / large model download.
    """

    CYBERBULLYING_CLASSES = [
        "age", "ethnicity", "gender",
        "religion", "not_cyberbullying", "other_cyberbullying",
    ]

    # Labels from unitary/toxic-bert that represent toxicity dimensions
    _TOXIC_BERT_LABELS = {
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
    }

    # Default model: use fine-tuned local model if available, else toxic-bert
    _DEFAULT_MODEL = str(MODEL_DIR)
    _FALLBACK_MODEL = "unitary/toxic-bert"

    def __init__(self, model_name: str = None):
        # Auto-select: prefer fine-tuned local model
        if model_name is None:
            import os
            if os.path.isdir(self._DEFAULT_MODEL):
                model_name = self._DEFAULT_MODEL
                logger.info("Using fine-tuned model: %s", model_name)
            else:
                model_name = self._FALLBACK_MODEL
                logger.info("Fine-tuned model not found, using: %s", model_name)
        self.model_name        = model_name
        self._use_transformers = False
        self._is_multilabel    = False   # True for toxic-bert style multi-label models
        self._model            = None
        self._tokenizer        = None
        self._try_load_transformers()

    def _try_load_transformers(self):
        """Attempt to load HuggingFace model; gracefully fall back if unavailable."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            logger.info("Loading BERT model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model     = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()
            self._use_transformers = True

            # Detect model type: multi-label (toxic-bert) vs multi-class
            all_labels = list(self._model.config.id2label.values())
            self._is_multilabel = any(
                lbl.lower() in self._TOXIC_BERT_LABELS for lbl in all_labels
            )
            mode = "multi-label sigmoid" if self._is_multilabel else "multi-class softmax"
            logger.info("BERT model loaded. Labels: %s | Mode: %s", all_labels, mode)

        except Exception as exc:
            logger.warning(
                "BERT model unavailable (%s). Using TF-Heuristic fallback.", exc
            )

    def predict(self, text: str) -> Tuple[float, str]:
        """
        Returns:
            p_toxic   : float – calibrated toxicity score (0–1)
            pred_class: str   – cyberbullying category (best guess)
        """
        if self._use_transformers:
            return self._predict_with_bert(text)
        return self._predict_heuristic(text)

    def _predict_with_bert(self, text: str) -> Tuple[float, str]:
        """
        BERT inference path.

        For multi-label models (unitary/toxic-bert):
          - Uses sigmoid (independent per-label probabilities)
          - p_toxic = max of all toxic-dimension scores, then temperature-scaled
          - Avoids saturation at 1.0 for ambiguous content

        For multi-class models:
          - Uses softmax, standard argmax classification
        """
        import torch
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits

        labels = self._model.config.id2label

        if self._is_multilabel:
            # ── Multi-label path (unitary/toxic-bert) ─────────────────────────
            # sigmoid gives independent probability per toxic dimension
            scores = torch.sigmoid(logits).squeeze()
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            scores_np = scores.numpy()

            # Map label index → score, keep only toxic-dimension labels
            toxic_scores: Dict[str, float] = {}
            for idx, lbl in labels.items():
                if lbl.lower() in self._TOXIC_BERT_LABELS:
                    toxic_scores[lbl] = float(scores_np[idx])

            if toxic_scores:
                dominant = max(toxic_scores, key=toxic_scores.get)
                raw_p    = max(toxic_scores.values())
            else:
                dominant = "toxic"
                raw_p    = float(scores_np.max())

            # Temperature scaling: softens extreme probabilities
            # Prevents benign-but-blunt text from always scoring 1.0
            p_toxic    = self._temperature_scale(raw_p, temperature=2.8)
            pred_class = self._map_toxic_label_to_cb_type(dominant, text)

        else:
            # ── Multi-class path (fine-tuned cyberbullying model) ─────────────
            # Apply temperature scaling to logits BEFORE softmax for calibration
            # T=2.5 spreads probabilities → prevents saturation on borderline text
            TEMPERATURE = 3.0   # Calibrated for fine-tuned cyberbullying model
            scaled_logits = logits / TEMPERATURE
            probs      = torch.softmax(scaled_logits, dim=-1).squeeze().numpy()
            pred_idx   = int(np.argmax(probs))
            pred_class = labels[pred_idx]

            # p_toxic = 1 - P(not_cyberbullying) after temperature scaling
            # Temperature scaling calibrates overconfident softmax outputs:
            #   "proposal could be improved" → P(other_cb) drops from 0.80 to 0.55
            #   "kill yourself"              → P(other_cb) stays high ~0.90
            not_cb_idx = next(
                (i for i, lbl in labels.items() if "not_cyberbullying" in lbl.lower()),
                None,
            )
            if not_cb_idx is not None:
                p_not_cb = float(probs[not_cb_idx])
                p_toxic  = float(1.0 - p_not_cb)
            else:
                p_toxic = float(sum(
                    probs[i] for i, lbl in labels.items()
                    if "not" not in lbl.lower()
                ))

        return float(np.clip(p_toxic, 0.0, 1.0)), pred_class

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _temperature_scale(prob: float, temperature: float = 2.8) -> float:
        """
        Soften a raw sigmoid probability via temperature scaling on its logit.

            logit  = ln( p / (1-p) )
            scaled = sigmoid( logit / T )

        T > 1 compresses toward 0.5, preventing saturation at 1.0 for
        blunt-but-not-critical text while keeping critical content high.
        """
        eps   = 1e-7
        prob  = float(np.clip(prob, eps, 1.0 - eps))
        logit = float(np.log(prob / (1.0 - prob)))
        return float(1.0 / (1.0 + np.exp(-logit / temperature)))

    # ── Keyword taxonomy for fine-grained classification ─────────────────────
    _CB_TYPE_KEYWORDS: Dict[str, List[str]] = {
        "religion": [
            "muslim", "islam", "islamic", "quran", "allah", "mosque",
            "jew", "jewish", "judaism", "hebrew", "synagogue", "zionist",
            "christian", "christianity", "church", "bible", "jesus", "christ", "pope",
            "hindu", "hinduism", "temple", "brahmin",
            "buddhist", "buddhism",
            "atheist", "atheism", "secular",
            "religion", "religious", "faith", "pray", "prayer", "worship",
            "infidel", "kafir", "blasphemy", "heathen", "cult",
            "sharia", "jihad", "crusade",
        ],
        "ethnicity": [
            # racial groups
            "black", "white", "asian", "hispanic", "latino", "latina",
            "arab", "arabic", "middle eastern", "african", "african american",
            "chinese", "japanese", "korean", "indian", "mexican", "puerto rican",
            # identity / origin terms
            "race", "racial", "racism", "racist", "ethnic", "ethnicity",
            "immigrant", "immigration", "illegal alien", "foreigner", "migrant",
            "native", "indigenous", "minority",
            # slurs (censored placeholders — replace with full lexicon)
            "nigger", "nigga", "spic", "chink", "gook", "wetback",
            "cracker", "kike", "raghead", "towelhead",
            # coded language
            "go back to your country", "not from here", "your people",
            "those people", "their kind",
        ],
        "gender": [
            # gender identity
            "woman", "women", "girl", "girls", "female", "females",
            "man", "men", "boy", "boys", "male", "males",
            "non-binary", "nonbinary", "enby",
            "trans", "transgender", "transwoman", "transman",
            "gender", "gendered", "genderfluid",
            # sexuality
            "gay", "lesbian", "bisexual", "bi", "homosexual",
            "queer", "lgbtq", "lgbtqia",
            # sexist / misogynist language
            "feminist", "feminism", "misogyn", "sexist", "sexism",
            "bitch", "whore", "slut", "cunt", "hoe",
            "mansplain", "simp", "incel",
            # stereotyping phrases
            "emotional", "too emotional", "belongs in the kitchen",
            "go make me a sandwich",
        ],
        "age": [
            "old", "older", "elderly", "senior", "ancient", "aged",
            "young", "younger", "teenager", "teen", "kid", "child",
            "boomer", "baby boomer", "gen x", "millennial", "gen z",
            "zoomer", "gen alpha",
            "grandpa", "grandma", "grandparent", "gramps",
            "immature", "childish", "grow up", "too young", "too old",
            "retire", "retirement", "past your prime",
            "age", "ageism", "ageist",
        ],
        "sexual_explicit": [
            # Direct solicitation / coercion
            "send nudes", "send me pictures", "show me your body",
            "i want to f", "want to f you", "i'll f you",
            # Sexual harassment phrases
            "you're so hot", "you're sexy", "i want you sexually",
            "sexually harass", "sexual harassment", "grope", "molest",
            # Non-consensual content
            "rape", "i will rape", "i want to rape", "non-consensual",
            # Explicit solicitation
            "suck my", "lick my", "touch yourself",
            # Degrading sexual language
            "sex object", "just good for sex", "only good for sex",
            "sleep with me", "have sex with me", "get in bed with me",
            # Explicit slurs used in sexual context
            "cum", "dick pic", "pussy", "cock",
        ],
    }

    @classmethod
    def _map_toxic_label_to_cb_type(cls, toxic_label: str, text: str) -> str:
        """
        Map a toxic-bert output label to the Kaggle cyberbullying taxonomy.

        Strategy (priority order):
          1. Score text against all keyword banks → pick highest match
          2. Use toxic_label as a tiebreaker for identity_hate
          3. Default to other_cyberbullying
        """
        text_lower = text.lower()

        # ── Score each category ───────────────────────────────────────────────
        scores: Dict[str, int] = {}
        for category, keywords in cls._CB_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score

        # ── Return best match if any keywords found ───────────────────────────
        if scores:
            return max(scores, key=scores.get)

        # ── Fallback: use toxic_label signal for identity_hate ────────────────
        if toxic_label.lower() == "identity_hate":
            return "other_cyberbullying"

        # ── threat / insult / obscene / toxic / severe_toxic ──────────────────
        return "other_cyberbullying"

    # ── TF-Heuristic fallback (used when transformers not installed) ──────────

    # Keyword banks grouped by cyberbullying type
    _KEYWORD_BANKS: Dict[str, List[str]] = {
        "age": [
            "old", "boomer", "millennial", "too young", "too old",
            "ancient", "grandpa", "kid", "immature", "childish",
        ],
        "ethnicity": [
            "race", "ethnic", "immigrant", "foreigner", "native",
            "white", "black", "asian", "hispanic", "arab",
            "go back", "your country",
        ],
        "gender": [
            "woman", "man", "girl", "boy", "female", "male",
            "feminist", "misogyn", "sexist", "trans", "gay", "lesbian",
        ],
        "religion": [
            "muslim", "christian", "jew", "hindu", "buddhist",
            "atheist", "infidel", "cult", "blasphemy", "heathen",
            "god", "allah", "bible", "quran",
        ],
        "other_cyberbullying": [
            "stupid", "idiot", "loser", "ugly", "fat", "dumb",
            "worthless", "kill yourself", "kys", "die", "hate you",
            "disgusting", "pathetic", "freak", "moron", "trash",
            "bully", "harass", "threat",
        ],
        "sexual_explicit": [
            # Solicitation
            "send nudes", "show me your body", "send me pictures",
            # Explicit acts
            "rape", "grope", "molest", "non-consensual",
            # Harassment phrases
            "sexual harassment", "sexually harass",
            "sleep with me", "have sex with me",
            "suck my", "lick my",
            # Degrading
            "sex object", "only good for sex", "just good for sex",
        ],
    }

    # Severity multipliers
    _HIGH_SEVERITY_TERMS = {
        "kill", "die", "hate", "ugly", "fat", "trash",
        "kys", "worthless", "disgusting", "pathetic",
        # Sexual severity terms
        "rape", "grope", "molest",
    }

    def _predict_heuristic(self, text: str) -> Tuple[float, str]:
        """Rule-of-thumb toxicity scorer used as BERT stand-in."""
        text_lower = text.lower()
        tokens     = set(re.findall(r"\b\w+\b", text_lower))
        scores: Dict[str, float] = {}

        for category, keywords in self._KEYWORD_BANKS.items():
            matches   = sum(1 for kw in keywords if kw in text_lower)
            severity  = sum(1.5 for kw in keywords if kw in tokens and kw in self._HIGH_SEVERITY_TERMS)
            scores[category] = matches + severity

        total = sum(scores.values())
        if total == 0:
            return 0.05, "not_cyberbullying"

        # Normalise to probability
        best_class = max(scores, key=scores.get)
        p_toxic    = min(total / (total + 4.0), 0.99)   # sigmoid-like squash
        return round(p_toxic, 4), best_class


# ── 3B. Rule-Based Algorithm  (§3.5.2) ──────────────────────────────────────

class RuleBasedDetector:
    """
    Deterministic detection of cyberbullying via:
      - Regular expressions for structural patterns
      - Curated abusive lexicons
      - High-risk direct-threat patterns  (binary flag)
      - Indirect bias / stereotyping patterns  (soft score 0–1)

    Output:
      detect()          → binary flag r_rule ∈ {0, 1}  (explicit abuse)
      detect_soft()     → float score ∈ [0, 1]          (weighted all signals)
    """

    # ── Explicit Slur Lexicon ─────────────────────────────────────────────────
    _EXPLICIT_SLURS = {
        # ethnic/racial
        "cracker", "spic", "chink", "wetback", "kike", "raghead",
        # gender / sexuality
        "bitch", "whore", "slut", "cunt",
        # general insults
        "idiot", "moron", "retard", "imbecile",
    }

    # ── High-Risk Phrases (direct threats / severe abuse) ─────────────────────
    _HIGH_RISK_PHRASES = [
        # Direct threats
        "kill yourself", "kys", "go die", "should die", "want you dead",
        "nobody likes you", "nobody loves you", "nobody wants you",
        "you are worthless", "worthless piece", "you are useless",
        "you are disgusting", "you are pathetic", "you are ugly",
        "i hate you", "i despise you", "i loathe you",
        "stab you", "burn in hell", "i will kill you", "i will hurt you",
        # Sexual threats / harassment (always explicit — binary flag)
        "i will rape you", "i want to rape", "send me nudes", "send nudes",
        "i'll rape", "going to rape", "sexually assault",
        "show me your body", "take off your clothes",
    ]


    # ── Indirect Bias Patterns (stereotyping, generalisation, micro-aggressions)
    # Weights are intentionally LOW to avoid false HIGH classifications
    # Only ACCUMULATION of multiple signals → significant score
    _INDIRECT_BIAS_PATTERNS = [
        # National / group generalisation (moderate weight)
        ("(all|those|these|every) (those )?(people|guys|folks|immigrants|foreigners) (are|from)", 0.25, "group_generalisation"),
        ("(people|guys|folks) from that (country|place|region|culture)",                          0.25, "national_generalisation"),
        # Ethnic / racial stereotype (higher weight — more explicit)
        ("(all|every) [a-z]+ are (stupid|lazy|criminals|terrorists|violent|dangerous|annoying)",  0.40, "ethnic_stereotype"),
        # Rhetorical bias
        ("why are (all )?(those|these|they) (people|guys)",                                        0.20, "rhetorical_bias"),
        ("(those|these) (people|immigrants|foreigners|migrants) (always|never|just|only)",         0.25, "coded_bias"),
        # Gender stereotype (LOW weight — avoid false HIGH)
        ("(women|girls|females?) are (just )?(too|so|very|always|never) [a-z]+",                  0.18, "gender_stereotype"),
        ("(men|boys|males?) are (just )?(too|so|very|always|never) [a-z]+",                       0.15, "gender_stereotype"),
        # Age stereotype (LOW weight)
        ("(boomers?|millennials?|gen z|zoomers?|old people|young people) are (all )?[a-z]+",      0.15, "age_stereotype"),
        # Religious stereotype (moderate)
        ("(muslims?|christians?|jews?|hindus?|atheists?) are (all )?(just )?[a-z]+",              0.25, "religion_stereotype"),
        # Group contempt
        ("(they|those people|these people) (are|is) (so )?(annoying|terrible|horrible|awful|stupid)", 0.25, "group_contempt"),
    ]

    def __init__(self):
        # High-risk: plain string matching (case-insensitive)
        self._high_risk_phrases = [p.lower() for p in self._HIGH_RISK_PHRASES]
        # Bias patterns: compile as simple regex (no word-boundary issues)
        self._compiled_bias = [
            (re.compile(p, re.IGNORECASE), w, lbl)
            for p, w, lbl in self._INDIRECT_BIAS_PATTERNS
        ]

    def detect(self, text: str) -> int:
        """Return 1 if EXPLICIT abusive pattern detected (direct threats / slurs), else 0."""
        text_lower = text.lower()

        # Check plain high-risk phrases
        for phrase in self._high_risk_phrases:
            if phrase in text_lower:
                logger.debug("Rule-match [explicit]: '%s'", phrase)
                return 1

        # Check slur lexicon
        tokens = set(re.findall(r"\b\w+\b", text_lower))
        if tokens & self._EXPLICIT_SLURS:
            logger.debug("Rule-match [slur]: explicit slur detected")
            return 1

        return 0


    def detect_soft(self, text: str) -> float:
        """
        Return a soft abuse score ∈ [0, 1] that captures both explicit
        and indirect / stereotyping patterns.

        Used by the Decision Engine as a supplementary signal.
        """
        text_lower = text.lower()

        # Explicit abuse = full score
        if self.detect(text_lower):
            return 1.0

        # Accumulate weighted indirect-bias matches
        score = 0.0
        for pattern, weight, label in self._compiled_bias:
            if pattern.search(text_lower):
                logger.debug("Rule-match [indirect, %s]: weight=%.2f", label, weight)
                score += weight

        return float(min(score, 1.0))


# ── 3C. Context Analysis Algorithm  (§3.5.3) ────────────────────────────────

class ContextAnalyzer:
    """
    Evaluates contextual signals to produce a normalised mitigation score
    C_context ∈ [0, 1].

    Higher score = more mitigating cues → lower final severity.
    """

    _POLITENESS_MARKERS = [
        r"\bplease\b", r"\bthank(s| you)\b", r"\brespectfully\b",
        r"\bi appreciate\b", r"\bi understand\b", r"\bi hope\b",
        r"\bkindly\b", r"\bsorry\b", r"\bexcuse\s+me\b",
        r"\bpardon\b", r"\bwith\s+all\s+due\s+respect\b",
    ]

    _NEUTRAL_DISCOURSE = [
        r"\bin\s+my\s+(opinion|view)\b",
        r"\bfrom\s+my\s+perspective\b",
        r"\bi\s+(think|believe|feel)\s+that\b",
        r"\bone\s+could\s+argue\b",
        r"\bsome\s+might\s+say\b",
        r"\baccording\s+to\b",
        r"\bstatistically\b", r"\bresearch\s+shows\b",
    ]

    _QUESTION_SOFTENERS = [
        r"\bcould\s+you\b", r"\bwould\s+you\b", r"\bmight\s+it\b",
        r"\bis\s+it\s+possible\b",
    ]

    def __init__(self):
        self._politeness_re  = [re.compile(p, re.IGNORECASE) for p in self._POLITENESS_MARKERS]
        self._neutral_re     = [re.compile(p, re.IGNORECASE) for p in self._NEUTRAL_DISCOURSE]
        self._question_re    = [re.compile(p, re.IGNORECASE) for p in self._QUESTION_SOFTENERS]

    def score(self, text: str) -> float:
        """Return C_context ∈ [0, 1]."""
        signals = 0
        max_signals = 5  # normalisation ceiling

        signals += sum(1 for p in self._politeness_re if p.search(text))
        signals += sum(0.7 for p in self._neutral_re   if p.search(text))
        signals += sum(0.5 for p in self._question_re  if p.search(text))

        # Sentence-level tone heuristic: question marks soften intent
        q_marks = text.count("?")
        signals += min(q_marks * 0.3, 0.9)

        return round(min(signals / max_signals, 1.0), 4)


# ── 3D. Constructive Criticism Mitigation  (§3.5.4) ──────────────────────────

class ConstructiveCriticismMitigator:
    """
    Detects linguistic patterns that indicate criticism directed at ideas
    rather than individuals.

    Output: binary flag mitigation_flag ∈ {True, False}
    """

    _IDEA_FOCUS_PATTERNS = [
        r"\bthe\s+(idea|concept|proposal|argument|point|claim)\b",
        r"\bthis\s+(approach|method|solution|plan|strategy)\b",
        r"\byour\s+(work|project|code|writing|approach|idea)\b",
        r"\bit\s+(could|should|might)\s+be\s+(improved|better|enhanced)\b",
        r"\bconstructive(ly)?\b",
        r"\bfeedback\b",
        r"\bi\s+suggest\b", r"\bsuggestion\b",
        r"\bperhaps\s+consider\b",
        r"\bone\s+way\s+to\s+improve\b",
    ]

    _RESPECT_SIGNALS = [
        r"\bbut\s+i\s+respect\b",
        r"\bno\s+offence\b", r"\bno\s+offense\b",
        r"\bnot\s+(targeting|attacking)\s+you\b",
        r"\bnothing\s+personal\b",
        r"\bwith\s+respect\b",
        r"\bmeaning\s+no\s+harm\b",
    ]

    def __init__(self):
        self._idea_re    = [re.compile(p, re.IGNORECASE) for p in self._IDEA_FOCUS_PATTERNS]
        self._respect_re = [re.compile(p, re.IGNORECASE) for p in self._RESPECT_SIGNALS]

    def is_constructive(self, text: str) -> bool:
        """Return True if text shows constructive intent, else False."""
        idea_match    = any(p.search(text) for p in self._idea_re)
        respect_match = any(p.search(text) for p in self._respect_re)
        return idea_match or respect_match


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4: INTEGRATION & DECISION LAYER  —  Decision Engine  (§3.5.5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionConfig:
    """
    Empirically optimised weights and thresholds for the Decision Engine.

    Severity formula  (§3.5.5, extended):
        S = w1 * p_toxic  +  w2 * max(r_rule, r_soft)  −  w3 * C_context

    With temperature-scaled BERT (T=2.8), expected p_toxic ranges:
        Clear abuse (logit ~6)   → p_toxic ≈ 0.88
        Moderate bias (logit ~3) → p_toxic ≈ 0.74
        Borderline (logit ~1)    → p_toxic ≈ 0.60
        Non-toxic (logit ~-2)    → p_toxic ≈ 0.40

    Threat thresholds (τ) — calibrated for scaled outputs:
        τ_c  =  0.65  → CRITICAL
        τ_h  =  0.42  → HIGH
        τ_l  =  0.20  → LOW
        else           → NOT_CB
    """
    w1:                   float = 0.45   # BERT toxicity weight   (calibrated for fine-tuned)
    w2:                   float = 0.30   # Rule-based soft weight
    w3:                   float = 0.25   # Context mitigation weight
    tau_c:                float = 0.62   # Critical threshold  (calibrated for fine-tuned model)
    tau_h:                float = 0.45   # High threshold     (calibrated for fine-tuned model)
    tau_l:                float = 0.34   # Low threshold
    mitigation_reduction: float = 0.45   # Strong reduction for constructive flag


class DecisionEngine:
    """
    Integrates all component outputs into a final ThreatLevel (§3.5.5).

    Steps:
      1. Fuse scores into severity S
      2. Apply constructive-criticism mitigation reduction if flagged
      3. Map S to ThreatLevel via thresholds
      4. Determine system actions
    """

    def __init__(self, config: Optional[DecisionConfig] = None):
        self.cfg = config or DecisionConfig()

    def decide(
        self,
        p_toxic:              float,
        r_rule:               int,
        r_rule_soft:          float,
        c_context:            float,
        mitigation_triggered: bool,
    ) -> Tuple[float, ThreatLevel, List[str]]:
        """
        Returns (severity_score, threat_level, actions).
        """
        # ── Step 7: Decision Fusion & Severity Computation ──────────────────
        # Override: explicit death threats / violence → always CRITICAL
        # (handles edge case where fine-tuned model underestimates direct abuse)
        if r_rule == 1:
            severity = 1.0
        else:
            # Cap r_soft at 0.50 — indirect bias patterns alone cannot push
            # a text into CRITICAL; only explicit hate + BERT together can
            r_soft_capped = min(r_rule_soft, 0.50)
            # Formula:  S = w1*p_toxic + w2*r_soft_capped - w3*C_context  (§3.5.5)
            severity = (
                self.cfg.w1 * p_toxic
                + self.cfg.w2 * r_soft_capped
                - self.cfg.w3 * c_context
            )

        # ── Step 6 effect: Constructive Criticism Mitigation ─────────────────
        if mitigation_triggered:
            severity -= self.cfg.mitigation_reduction

        if r_rule == 0 and c_context >= 0.20 and severity < 0.38:
            severity = 0.0

        severity = float(np.clip(severity, 0.0, 1.0))

        # ── Step 8: Threshold-Based Classification ───────────────────────────
        if severity >= self.cfg.tau_c:
            level = ThreatLevel.CRITICAL
        elif severity >= self.cfg.tau_h:
            level = ThreatLevel.HIGH
        elif severity >= self.cfg.tau_l:
            level = ThreatLevel.LOW
        else:
            level = ThreatLevel.NOT_CB

        # ── Step 9: System Actions ────────────────────────────────────────────
        actions = self._determine_actions(level)

        return round(severity, 4), level, actions

    @staticmethod
    def _determine_actions(level: ThreatLevel) -> List[str]:
        """Map threat level to system response actions."""
        action_map = {
            ThreatLevel.NOT_CB:   ["log"],
            ThreatLevel.LOW:      ["log", "flag_for_review"],
            ThreatLevel.HIGH:     ["log", "notify_moderator", "update_dashboard"],
            ThreatLevel.CRITICAL: ["log", "notify_moderator", "update_dashboard",
                                   "alert_user", "escalate_to_human_review"],
        }
        return action_map[level]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SYSTEM  —  Orchestrates All 9 Steps
# ─────────────────────────────────────────────────────────────────────────────

class IntegratedCyberbullyingDetectionSystem:
    """
    Full hybrid detection pipeline as described in §3.5 and Figure 3.1.

    Usage:
        system = IntegratedCyberbullyingDetectionSystem()
        result = system.analyze("Your text here")
        print(result.threat_level, result.severity_score)
    """

    def __init__(self, decision_config: Optional[DecisionConfig] = None, bert_analyzer: Optional[BERTToxicityAnalyzer] = None):
        logger.info("Initialising Integrated Cyberbullying Detection System...")
        self.preprocessor  = TextPreprocessor()
        self.bert_analyzer = bert_analyzer or BERTToxicityAnalyzer()
        self.rule_detector = RuleBasedDetector()
        self.context_analyzer = ContextAnalyzer()
        self.mitigator     = ConstructiveCriticismMitigator()
        self.decision_engine = DecisionEngine(decision_config)
        logger.info("System ready.")

    # ── public interface ──────────────────────────────────────────────────────

    def analyze(self, text: str) -> DetectionResult:
        """
        Run the full 9-step detection pipeline on a single text input.
        """
        # ── Step 1: Input Acquisition ──────────────────────────────────────
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string.")

        # ── Step 2: Text Preprocessing ─────────────────────────────────────
        clean_text = self.preprocessor.preprocess(text)

        # ── Step 3: Transformer-Based Toxicity Analysis ─────────────────────
        p_toxic, predicted_type = self.bert_analyzer.predict(clean_text)

        # ── Step 4: Rule-Based Abuse Detection ──────────────────────────────
        r_rule      = self.rule_detector.detect(clean_text)
        r_rule_soft = self.rule_detector.detect_soft(clean_text)

        # ── Step 5: Context Analysis ─────────────────────────────────────────
        c_context = self.context_analyzer.score(clean_text)

        # ── Step 6: Constructive Criticism Mitigation ────────────────────────
        mitigation_triggered = self.mitigator.is_constructive(clean_text)

        # ── Steps 7-9: Decision Fusion → Classification → Actions ────────────
        severity, threat_level, actions = self.decision_engine.decide(
            p_toxic, r_rule, r_rule_soft, c_context, mitigation_triggered
        )

        return DetectionResult(
            text=text,
            p_toxic=p_toxic,
            r_rule=r_rule,
            r_rule_soft=r_rule_soft,
            c_context=c_context,
            mitigation_triggered=mitigation_triggered,
            severity_score=severity,
            threat_level=threat_level,
            predicted_type=predicted_type,
            actions=actions,
        )

    def analyze_batch(self, texts: List[str]) -> List[DetectionResult]:
        """Analyze a list of texts and return results."""
        results = []
        for i, text in enumerate(texts):
            try:
                results.append(self.analyze(text))
            except Exception as exc:
                logger.warning("Failed to analyze text[%d]: %s", i, exc)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION MODULE  (§3.7)
# ─────────────────────────────────────────────────────────────────────────────

class SystemEvaluator:
    """
    Computes Precision, Recall, and F1-Score as described in §3.7.

    Metrics formulas:
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1-score  = 2 · (Precision · Recall) / (Precision + Recall)
    """

    @staticmethod
    def evaluate(
        y_true: List[str],
        y_pred: List[str],
    ) -> Dict[str, float]:
        """
        Compute macro-averaged Precision, Recall, F1 over all classes.

        Parameters:
            y_true : ground-truth labels
            y_pred : predicted labels

        Returns:
            dict with keys: precision, recall, f1, accuracy
        """
        from collections import defaultdict

        classes = sorted(set(y_true) | set(y_pred))
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        for t, p in zip(y_true, y_pred):
            if t == p:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1

        precision_per_class, recall_per_class, f1_per_class = {}, {}, {}
        for cls in classes:
            prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
            rec  = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            precision_per_class[cls] = prec
            recall_per_class[cls]    = rec
            f1_per_class[cls]        = f1

        macro_precision = np.mean(list(precision_per_class.values()))
        macro_recall    = np.mean(list(recall_per_class.values()))
        macro_f1        = np.mean(list(f1_per_class.values()))
        accuracy        = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

        return {
            "accuracy":  round(accuracy,        4),
            "precision": round(macro_precision,  4),
            "recall":    round(macro_recall,     4),
            "f1":        round(macro_f1,         4),
            "per_class": {
                cls: {
                    "precision": round(precision_per_class[cls], 4),
                    "recall":    round(recall_per_class[cls],    4),
                    "f1":        round(f1_per_class[cls],        4),
                }
                for cls in classes
            },
        }

    @staticmethod
    def confusion_matrix(
        y_true: List[str],
        y_pred: List[str],
    ) -> pd.DataFrame:
        """Return a labelled confusion matrix as a DataFrame."""
        classes = sorted(set(y_true) | set(y_pred))
        matrix  = pd.DataFrame(0, index=classes, columns=classes)
        for t, p in zip(y_true, y_pred):
            matrix.loc[t, p] += 1
        return matrix


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER  —  Kaggle Cyberbullying Classification Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads and prepares the Kaggle Cyberbullying Classification Dataset (§3.3).
    Supports CSV, XLSX, and TSV formats.
    """

    @staticmethod
    def load(path: str, sample_n: Optional[int] = None) -> pd.DataFrame:
        """
        Load dataset from file.

        Parameters:
            path     : path to the data file
            sample_n : if set, return a random sample of this size

        Returns:
            DataFrame with columns ['tweet_text', 'cyberbullying_type']
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)

        # Standardise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        if "tweet_text" not in df.columns:
            raise ValueError(
                f"Expected column 'tweet_text', found: {list(df.columns)}"
            )

        df = df.dropna(subset=["tweet_text"])
        df["tweet_text"] = df["tweet_text"].astype(str)

        if sample_n:
            df = df.sample(n=min(sample_n, len(df)), random_state=42)

        logger.info(
            "Dataset loaded: %d rows, distribution:\n%s",
            len(df),
            df["cyberbullying_type"].value_counts().to_string(),
        )
        return df

    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified train/test split."""
        from sklearn.model_selection import train_test_split as _split
        train, test = _split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["cyberbullying_type"],
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# BERT FINE-TUNING MODULE  (§3.5.1 — for production training)
# ─────────────────────────────────────────────────────────────────────────────

class BERTFineTuner:
    """
    Fine-tunes bert-base-uncased on the Kaggle cyberbullying dataset.
    Requires: transformers, torch, sklearn

    Usage:
        tuner = BERTFineTuner()
        tuner.train(train_df)
        tuner.save("./cyberbullying_bert")
    """

    MODEL_NAME    = "bert-base-uncased"
    MAX_LENGTH    = 128
    BATCH_SIZE    = 16
    NUM_EPOCHS    = 3
    LEARNING_RATE = 2e-5
    LABEL_MAP     = {
        "age":                  0,
        "ethnicity":            1,
        "gender":               2,
        "religion":             3,
        "not_cyberbullying":    4,
        "other_cyberbullying":  5,
    }

    def __init__(self):
        self._check_dependencies()
        self._model = None       # set in train()
        self._tokenizer = None   # set in train()

    @staticmethod
    def _check_dependencies():
        try:
            import torch
            import transformers
            import sklearn
        except ImportError as e:
            raise ImportError(
                f"Fine-tuning requires: torch, transformers, sklearn — missing: {e}"
            )

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """
        Fine-tune BERT for sequence classification.

        Parameters:
            train_df : DataFrame with ['tweet_text', 'cyberbullying_type']
            val_df   : optional validation split
        """
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW  # AdamW moved to torch.optim in newer versions
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )

        bert_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=len(self.LABEL_MAP),
            id2label={v: k for k, v in self.LABEL_MAP.items()},
            label2id=self.LABEL_MAP,
        )

        class CyberbullyingDataset(Dataset):
            def __init__(self, df, tokenizer, label_map, max_len):
                self.texts  = df["tweet_text"].tolist()
                self.labels = [label_map.get(lbl, 4) for lbl in df["cyberbullying_type"]]
                self.tok    = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = self.tok(
                    self.texts[idx],
                    truncation=True,
                    max_length=self.max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids":      enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
                }

        train_ds = CyberbullyingDataset(train_df, bert_tokenizer, self.LABEL_MAP, self.MAX_LENGTH)
        train_dl = DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True)

        # val_df can be used for validation during training if provided
        val_dl = None
        if val_df is not None:
            val_ds = CyberbullyingDataset(val_df, bert_tokenizer, self.LABEL_MAP, self.MAX_LENGTH)
            val_dl = DataLoader(val_ds, batch_size=self.BATCH_SIZE, shuffle=False)

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE)
        total_steps = len(train_dl) * self.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        def evaluate_loader(data_loader):
            model.eval()
            y_true, y_pred = [], []
            id2label = {v: k for k, v in self.LABEL_MAP.items()}
            with torch.no_grad():
                for batch in data_loader:
                    logits = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).logits
                    preds = torch.argmax(logits, dim=-1).cpu().tolist()
                    labels = batch["labels"].cpu().tolist()
                    y_pred.extend(id2label.get(p, "not_cyberbullying") for p in preds)
                    y_true.extend(id2label.get(l, "not_cyberbullying") for l in labels)
            model.train()
            return SystemEvaluator.evaluate(y_true, y_pred)

        logger.info("Starting BERT fine-tuning on device: %s", device)
        best_f1 = -1.0
        best_state = None
        for epoch in range(self.NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            for batch in train_dl:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dl)
            msg = f"Epoch {epoch + 1}/{self.NUM_EPOCHS} — Avg Loss: {avg_loss:.4f}"
            if val_dl is not None:
                metrics = evaluate_loader(val_dl)
                msg += f" | Val F1: {metrics['f1']:.4f} | Val Acc: {metrics['accuracy']:.4f}"
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info(msg)

        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info("Loaded best validation checkpoint (F1=%.4f).", best_f1)

        self._model     = model
        self._tokenizer = bert_tokenizer
        logger.info("Fine-tuning complete.")

    def save(self, output_dir: str):
        """Save fine-tuned model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        logger.info("Model saved to: %s", output_dir)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Run inference on test set and return evaluation metrics."""
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()

        from torch.utils.data import Dataset, DataLoader

        id2label = {v: k for k, v in self.LABEL_MAP.items()}
        tokenizer = self._tokenizer
        _max_length = self.MAX_LENGTH

        class EvalDataset(Dataset):
            def __init__(self, df):
                self.texts = df["tweet_text"].astype(str).tolist()
                self.labels = df["cyberbullying_type"].astype(str).tolist()

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = tokenizer(
                    self.texts[idx],
                    truncation=True,
                    max_length=_max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label": self.labels[idx],
                }

        y_true, y_pred = [], []
        loader = DataLoader(EvalDataset(test_df), batch_size=self.BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                logits = self._model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()
                y_true.extend(batch["label"])
                y_pred.extend(id2label.get(i, "not_cyberbullying") for i in pred_indices)

        return SystemEvaluator.evaluate(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# DEMONSTRATION  —  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result: DetectionResult):
    width = 70
    print("=" * width)
    print(f"  INPUT  : {result.text[:65]}{'...' if len(result.text)>65 else ''}")
    print("-" * width)
    print(f"  BERT p_toxic         : {result.p_toxic:.4f}")
    print(f"  Rule-Based flag      : {result.r_rule}  (soft: {result.r_rule_soft:.4f})")
    print(f"  Context score        : {result.c_context:.4f}")
    print(f"  Mitigation triggered : {result.mitigation_triggered}")
    print(f"  Predicted type       : {result.predicted_type}")
    print(f"  ── Severity Score    : {result.severity_score:.4f}")
    print(f"  ── Threat Level      : {result.threat_level.value}")
    print(f"  ── Actions           : {', '.join(result.actions)}")
    print("=" * width)


def run_demo_with_dataset(dataset_path: str, n_samples: int = 10):
    """
    Load the Kaggle dataset, run the detection pipeline on a sample,
    and display evaluation metrics.
    """
    print("\n" + "█" * 70)
    print("  Integrated AI Cybercrime Detection System — Demonstration")
    print("  Al-Balqa Applied University — Faculty of Artificial Intelligence")
    print("█" * 70 + "\n")

    # 1. Load dataset
    df = DatasetLoader.load(dataset_path, sample_n=n_samples)

    # 2. Initialise system
    system = IntegratedCyberbullyingDetectionSystem()

    # 3. Analyse each sample
    detection_results = []
    y_true  = []
    y_pred  = []

    print(f"\nAnalysing {len(df)} samples from the Kaggle dataset...\n")
    for _, row in df.iterrows():
        detection = system.analyze(row["tweet_text"])
        detection_results.append(detection)
        y_true.append(row["cyberbullying_type"])

        # Map threat level back to a comparable class
        if detection.threat_level == ThreatLevel.NOT_CB:
            y_pred.append("not_cyberbullying")
        elif detection.predicted_type:
            y_pred.append(detection.predicted_type)
        else:
            y_pred.append("other_cyberbullying")

        _print_result(detection)

    # 4. Aggregate evaluation
    eval_metrics = SystemEvaluator.evaluate(y_true, y_pred)
    print("\n" + "─" * 70)
    print("  EVALUATION METRICS  (§3.7)")
    print("─" * 70)
    print(f"  Accuracy  : {eval_metrics['accuracy']:.4f}")
    print(f"  Precision : {eval_metrics['precision']:.4f}")
    print(f"  Recall    : {eval_metrics['recall']:.4f}")
    print(f"  F1-Score  : {eval_metrics['f1']:.4f}")
    print("─" * 70)

    # 5. Threat level distribution
    threat_counts: Dict[str, int] = {}
    for r in detection_results:
        key = r.threat_level.value
        threat_counts[key] = threat_counts.get(key, 0) + 1
    print("\n  Threat Level Distribution:")
    for level, count in sorted(threat_counts.items()):
        bar = "█" * count
        print(f"    {level:<12}: {bar} ({count})")
    print()

    return detection_results, eval_metrics


def run_custom_examples():
    """Run the pipeline on hand-crafted illustrative examples."""
    system = IntegratedCyberbullyingDetectionSystem()

    examples = [
        # (label,  text)
        ("CRITICAL",  "You are worthless and should go die. Nobody likes you."),
        ("HIGH",      "I hate people of your religion, you're all idiots."),
        ("LOW",       "Women are just too emotional to be good leaders anyway."),
        ("NOT_CB",    "I think the proposal could be improved. The idea itself has merit."),
        ("NOT_CB",    "Respectfully, I disagree with your argument but I respect your view."),
        ("CRITIQUE",  "Your code has bugs, but I believe the overall approach is solid."),
        ("ETHNIC",    "Why are all those people from that country so annoying?"),
    ]

    print("\n" + "█" * 70)
    print("  CUSTOM EXAMPLE ANALYSIS")
    print("█" * 70)
    for expected, text in examples:
        custom_result = system.analyze(text)
        status = "✓" if expected in custom_result.threat_level.value else "✖"
        print(f"\n[Expected: {expected:<10}] [{status}]")
        _print_result(custom_result)



if __name__ == "__main__":
    DATASET_PATH = DATA_DIR / "cyberbullying_tweets.csv.xlsx"

    # Check if dataset is present
    if DATASET_PATH.is_file():
        demo_results, demo_metrics = run_demo_with_dataset(str(DATASET_PATH), n_samples=20)
    else:
        logger.warning(
            "Dataset not found at '%s'. Running custom examples only.", DATASET_PATH
        )
        run_custom_examples()

