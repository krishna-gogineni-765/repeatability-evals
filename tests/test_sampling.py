import itertools

import pytest

from repeatability_evals import AllPairs, AnchorPairs, CycleSampler, RandomPairs, cycle_pairs


def test_all_pairs():
    assert AllPairs().sample(4) == list(itertools.combinations(range(4), 2))
    assert AllPairs().sample(1) == []


@pytest.mark.parametrize("n", [3, 4, 5, 8, 10, 25])
def test_default_budget_is_a_cycle_covering_every_item_twice(n):
    pairs = cycle_pairs(n, seed=7)
    assert len(pairs) == min(n, n * (n - 1) // 2) == len(set(pairs))
    degree = {k: 0 for k in range(n)}
    for i, j in pairs:
        assert i < j
        degree[i] += 1
        degree[j] += 1
    assert all(d >= 2 for d in degree.values())


def test_budget_at_or_above_total_gives_exact_all_pairs():
    assert cycle_pairs(5, 10) == AllPairs().sample(5) == cycle_pairs(5, 999)


def test_small_budget_prefers_covering_many_items():
    pairs = cycle_pairs(8, 4, seed=1)
    assert len(pairs) == 4 and {x for p in pairs for x in p} == set(range(8))


def test_extra_budget_adds_distinct_pairs():
    pairs = cycle_pairs(10, 20, seed=3)
    assert len(pairs) == 20 == len(set(pairs))


def test_deterministic_and_seed_sensitive():
    assert cycle_pairs(10, 10, seed=5) == cycle_pairs(10, 10, seed=5)
    assert cycle_pairs(10, 10, seed=5) != cycle_pairs(10, 10, seed=6)


def test_edge_cases():
    assert cycle_pairs(0) == [] and cycle_pairs(1) == []
    assert cycle_pairs(2, 1) == [(0, 1)]
    with pytest.raises(ValueError):
        cycle_pairs(5, 0)


def test_cycle_sampler_class_matches_function():
    assert CycleSampler(budget=6, seed=2).sample(9) == cycle_pairs(9, 6, 2)


def test_random_pairs():
    pairs = RandomPairs(5, seed=1).sample(10)
    assert len(pairs) == 5 == len(set(pairs))
    assert RandomPairs(99).sample(4) == AllPairs().sample(4)


def test_anchor_pairs():
    assert AnchorPairs(0).sample(4) == [(0, 1), (0, 2), (0, 3)]
    assert AnchorPairs(2).sample(4) == [(0, 2), (1, 2), (2, 3)]
    assert AnchorPairs().sample(1) == []
    with pytest.raises(IndexError):
        AnchorPairs(9).sample(4)
