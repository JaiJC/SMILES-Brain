"""Tests for the generative proposal search and shared screening helpers."""

import numpy as np
import pytest

from smilesbrain.design.generate import (
    STATUS,
    compare_to_random,
    evolve,
    random_search,
)
from smilesbrain.design.screen import high_low_target, region_indices

ALPHABET = "abcdefgh"


def _toy_score(s: str) -> float:
    # Reward 'a' content; penalise length so the optimum is a short string of a's.
    return s.count("a") - 0.05 * len(s)


def _toy_mutate(s: str, rng: np.random.Generator) -> str | None:
    op = int(rng.integers(0, 3))
    ch = ALPHABET[int(rng.integers(0, len(ALPHABET)))]
    if op == 0 and s:
        i = int(rng.integers(0, len(s)))
        return s[:i] + ch + s[i + 1:]
    if op == 1:
        i = int(rng.integers(0, len(s) + 1))
        return s[:i] + ch + s[i:]
    return s[:-1] if len(s) > 1 else None


def test_evolve_is_reproducible_and_beats_random_search():
    seeds = ["bcdefgh", "hgfedcb"]
    gains = []
    for seed in range(5):
        a = evolve(seeds, _toy_score, _toy_mutate, rng=np.random.default_rng(seed),
                   population=16, generations=12)
        b = evolve(seeds, _toy_score, _toy_mutate, rng=np.random.default_rng(seed),
                   population=16, generations=12)
        assert [(p.smiles, p.score) for p in a.proposals] == [
            (p.smiles, p.score) for p in b.proposals
        ]
        base = random_search(seeds, _toy_score, _toy_mutate, rng=np.random.default_rng(seed + 100),
                             n_evaluations=a.n_evaluations)
        assert base.n_evaluations <= a.n_evaluations
        gains.append(compare_to_random(a, base, k=5)["improvement_top_k_mean"])
    assert np.mean(gains) > 0 and min(gains) > 0


def test_evolve_proposals_are_distinct_novel_and_ranked():
    seeds = ["bcdefgh"]
    res = evolve(seeds, _toy_score, _toy_mutate, rng=np.random.default_rng(1),
                 population=8, generations=5)
    smiles = [p.smiles for p in res.proposals]
    assert len(smiles) == len(set(smiles))
    scores = [p.score for p in res.proposals]
    assert scores == sorted(scores, reverse=True)
    assert all(p.generation > 0 for p in res.top(5))
    assert all(p.parent is not None for p in res.top(5))
    assert res.best_by_generation == sorted(res.best_by_generation)  # elitist: never worse


def test_evolve_rejects_unscorable_seeds_and_nonfinite_scores():
    with pytest.raises(ValueError):
        evolve(["x"], lambda s: float("nan"), _toy_mutate, rng=np.random.default_rng(0))
    res = evolve(["bcd"], lambda s: float("nan") if len(s) > 6 else _toy_score(s), _toy_mutate,
                 rng=np.random.default_rng(0), population=4, generations=4)
    assert all(np.isfinite(p.score) and len(p.smiles) <= 6 for p in res.proposals)


def test_status_label_is_explicit():
    assert "UNVALIDATED" in STATUS


def test_region_indices_and_target():
    assert region_indices("3, 1,3", 10) == [1, 3]
    assert region_indices(None, 10) == []
    with pytest.raises(ValueError):
        region_indices("10", 10)
    with pytest.raises(ValueError):
        region_indices("a", 10)
    target = high_low_target(5, [0], [4])
    assert target[0] == 1 and target[4] == -1 and np.isnan(target[1:4]).all()
    with pytest.raises(ValueError):
        high_low_target(5, [1], [1])


def test_rdkit_mutations_yield_valid_in_bounds_molecules():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from smilesbrain.design.generate import (
        MAX_HEAVY,
        MIN_HEAVY,
        rdkit_canonicalize,
        rdkit_mutate,
    )

    rng = np.random.default_rng(0)
    nicotine = "CN1CCC[C@H]1c1cccnc1"
    assert rdkit_canonicalize("not a smiles") is None
    assert rdkit_canonicalize("CC") is None  # below the heavy-atom floor
    assert rdkit_canonicalize("CCCCOOCCCC") is None  # peroxide
    assert rdkit_canonicalize("CCS(O)(c1ccccc1)CC") is None  # sulfur with no S=O
    assert rdkit_canonicalize("CCS(=O)(=O)c1ccccc1") is not None  # sulfone is fine
    assert rdkit_canonicalize("CCSc1ccccc1") is not None  # thioether is fine
    assert rdkit_canonicalize("CCO.CCCCCCCC") == rdkit_canonicalize("CCCCCCCC")  # largest frag
    produced = set()
    for _ in range(200):
        child = rdkit_mutate(nicotine, rng)
        if child is None:
            continue
        mol = Chem.MolFromSmiles(child)
        assert mol is not None and MIN_HEAVY <= mol.GetNumHeavyAtoms() <= MAX_HEAVY
        produced.add(child)
    assert len(produced) > 20  # the operator actually explores


def test_evolve_with_rdkit_improves_a_simple_property():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from smilesbrain.design.generate import rdkit_canonicalize, rdkit_mutate

    def nitrogen_fraction(smi: str) -> float:
        mol = Chem.MolFromSmiles(smi)
        return sum(a.GetAtomicNum() == 7 for a in mol.GetAtoms()) / mol.GetNumHeavyAtoms()

    seeds = ["CCCCCCCC", "c1ccccc1CC"]
    kw = dict(canonicalize=rdkit_canonicalize)
    res = evolve(seeds, nitrogen_fraction, rdkit_mutate, rng=np.random.default_rng(0),
                 population=16, generations=8, **kw)
    base = random_search(seeds, nitrogen_fraction, rdkit_mutate, rng=np.random.default_rng(1),
                         n_evaluations=res.n_evaluations, **kw)
    assert compare_to_random(res, base, k=5)["improvement_top_k_mean"] > 0
