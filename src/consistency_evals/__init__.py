"""consistency-evals: composable tools for measuring how consistent an LLM system's runs are."""
from .dataset import DatasetResult, aevaluate_groups, evaluate_groups
from .metrics import ClusterConsistency, ConsistencyMetric, PairwiseConsistency, TotalAgreement
from .prompts import SEMANTIC_EQUIVALENCE_PROMPT
from .sampling import AllPairs, AnchorPairs, CycleSampler, PairSampler, RandomPairs, cycle_pairs
from .similarity import (
    DiceSimilarity,
    EditSimilarity,
    EmbeddingCosine,
    ExactMatch,
    FunctionSimilarity,
    JaccardSimilarity,
    JudgeParseError,
    LLMJudge,
    NgramJaccard,
    OverlapCoefficient,
    RankBiasedOverlap,
    RougeL,
    Similarity,
    as_similarity,
)
from .stats import bootstrap_ci
from .types import ConsistencyResult, PairScore, SimilarityResult

__version__ = "0.1.0"

__all__ = [
    "ConsistencyMetric", "PairwiseConsistency", "ClusterConsistency", "TotalAgreement",
    "evaluate_groups", "aevaluate_groups", "DatasetResult", "bootstrap_ci",
    "Similarity", "FunctionSimilarity", "as_similarity",
    "JaccardSimilarity", "DiceSimilarity", "OverlapCoefficient",
    "ExactMatch", "NgramJaccard", "RougeL", "EditSimilarity",
    "RankBiasedOverlap", "EmbeddingCosine", "LLMJudge", "JudgeParseError",
    "PairSampler", "AllPairs", "CycleSampler", "RandomPairs", "AnchorPairs", "cycle_pairs",
    "ConsistencyResult", "PairScore", "SimilarityResult", "SEMANTIC_EQUIVALENCE_PROMPT",
]
