"""repeatability-evals: consistency evals for non-deterministic LLM systems."""
from .compare import InputComparison, VersionComparison, acompare_versions, compare_versions
from .dataset import DatasetResult, aevaluate_groups, evaluate_groups
from .metrics import ClusterConsistency, ConsistencyMetric, PairwiseConsistency, TotalAgreement
from .prompts import DEFAULT_DIMENSIONS, SEMANTIC_EQUIVALENCE_PROMPT, Dimension, build_rubric_prompt
from .sampling import AllPairs, AnchorPairs, CrossSampler, CycleSampler, PairSampler, RandomPairs, cycle_pairs
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
    RubricJudge,
    Similarity,
    as_similarity,
)
from .stats import bootstrap_ci
from .types import ConsistencyResult, PairScore, SimilarityResult

__version__ = "0.1.0"

__all__ = [
    "compare_versions", "acompare_versions", "VersionComparison", "InputComparison", "CrossSampler",
    "ConsistencyMetric", "PairwiseConsistency", "ClusterConsistency", "TotalAgreement",
    "evaluate_groups", "aevaluate_groups", "DatasetResult", "bootstrap_ci",
    "Similarity", "FunctionSimilarity", "as_similarity",
    "JaccardSimilarity", "DiceSimilarity", "OverlapCoefficient",
    "ExactMatch", "NgramJaccard", "RougeL", "EditSimilarity",
    "RankBiasedOverlap", "EmbeddingCosine", "LLMJudge", "RubricJudge", "JudgeParseError",
    "Dimension", "DEFAULT_DIMENSIONS", "build_rubric_prompt",
    "PairSampler", "AllPairs", "CycleSampler", "RandomPairs", "AnchorPairs", "cycle_pairs",
    "ConsistencyResult", "PairScore", "SimilarityResult", "SEMANTIC_EQUIVALENCE_PROMPT",
]
