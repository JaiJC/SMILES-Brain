"""Dependency-light generative proposal: a mutation-based genetic search over molecules.

Everything produced here is an **unvalidated hypothesis**. A proposal is only "good" with
respect to a scoring function, which in this project means: a predicted affinity vector that
matches receptor weights solved from a target regional profile (see `design.score`). The
validation work in `docs/VALIDATION.md` found that binding affinity is *not* currently a
validated predictor of drug-induced brain maps, so a high score is a statement about the
objective, not about biology. Proposals must never be presented as validated compounds.

The search loop (`evolve`) is independent of chemistry: it takes a `score_fn` and a
`mutate_fn` over strings, so it is testable without RDKit. `rdkit_mutate` is the default
chemical mutation operator (RDKit is imported lazily). Every random choice flows through the
`numpy.random.Generator` you pass in, so a fixed seed reproduces a run exactly.

`random_search` spends the same evaluation budget with no selection pressure; a proposal
method that cannot beat it is not doing anything useful, so `compare_to_random` reports both.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

STATUS = "UNVALIDATED HYPOTHESIS"

ScoreFn = Callable[[str], float]
MutateFn = Callable[[str, np.random.Generator], "str | None"]


@dataclass(frozen=True)
class Proposal:
    smiles: str
    score: float
    generation: int          # 0 = a seed molecule
    parent: str | None       # SMILES it was mutated from (None for seeds)


@dataclass
class SearchResult:
    proposals: list[Proposal]        # every distinct scored candidate, best first
    n_evaluations: int               # number of score_fn calls spent
    best_by_generation: list[float] = field(default_factory=list)

    def top(self, k: int, *, include_seeds: bool = False) -> list[Proposal]:
        pool = [p for p in self.proposals if include_seeds or p.generation > 0]
        return pool[:k]


def _rank(items: Iterable[Proposal]) -> list[Proposal]:
    # Deterministic: score descending, then SMILES ascending to break ties.
    return sorted(items, key=lambda p: (-p.score, p.smiles))


def _score(score_fn: ScoreFn, smiles: str) -> float | None:
    value = float(score_fn(smiles))
    return value if np.isfinite(value) else None


def _seed_pool(
    seeds: Iterable[str], score_fn: ScoreFn, canonicalize: Callable[[str], str | None]
) -> tuple[dict[str, Proposal], int]:
    pool: dict[str, Proposal] = {}
    n_eval = 0
    for raw in seeds:
        smi = canonicalize(raw)
        if smi is None or smi in pool:
            continue
        n_eval += 1
        value = _score(score_fn, smi)
        if value is not None:
            pool[smi] = Proposal(smi, value, 0, None)
    if not pool:
        raise ValueError("no seed molecule could be parsed and scored")
    return pool, n_eval


def _identity(smiles: str) -> str | None:
    return smiles


def evolve(
    seeds: Iterable[str],
    score_fn: ScoreFn,
    mutate_fn: MutateFn,
    *,
    rng: np.random.Generator,
    population: int = 32,
    generations: int = 10,
    offspring: int | None = None,
    tournament: int = 3,
    canonicalize: Callable[[str], str | None] = _identity,
    max_attempts: int = 5,
) -> SearchResult:
    """Tournament-selection mutation search; returns every distinct scored candidate."""
    if population < 1 or generations < 0 or tournament < 1:
        raise ValueError("population and tournament must be >= 1 and generations >= 0")
    offspring = population if offspring is None else offspring
    seen, n_eval = _seed_pool(seeds, score_fn, canonicalize)
    pop = _rank(seen.values())[:population]
    history = [pop[0].score]
    for gen in range(1, generations + 1):
        for _ in range(offspring):
            contenders = [pop[i] for i in rng.integers(0, len(pop), size=tournament)]
            parent = max(contenders, key=lambda p: (p.score, p.smiles))
            child = None
            for _attempt in range(max_attempts):
                mutated = mutate_fn(parent.smiles, rng)
                mutated = canonicalize(mutated) if mutated is not None else None
                if mutated is not None and mutated not in seen:
                    child = mutated
                    break
            if child is None:
                continue
            n_eval += 1
            value = _score(score_fn, child)
            if value is None:
                seen[child] = Proposal(child, -np.inf, gen, parent.smiles)  # remember, never keep
                continue
            seen[child] = Proposal(child, value, gen, parent.smiles)
        pop = _rank(p for p in seen.values() if np.isfinite(p.score))[:population]
        history.append(pop[0].score)
    ranked = _rank(p for p in seen.values() if np.isfinite(p.score))
    return SearchResult(ranked, n_eval, history)


def random_search(
    seeds: Iterable[str],
    score_fn: ScoreFn,
    mutate_fn: MutateFn,
    *,
    rng: np.random.Generator,
    n_evaluations: int,
    canonicalize: Callable[[str], str | None] = _identity,
    max_attempts: int = 5,
) -> SearchResult:
    """Baseline: same mutation operator and budget, parents chosen uniformly (no selection)."""
    seen, n_eval = _seed_pool(seeds, score_fn, canonicalize)
    parents = list(seen)
    stalled = 0
    while n_eval < n_evaluations and stalled < 50:
        parent = parents[int(rng.integers(0, len(parents)))]
        child = None
        for _attempt in range(max_attempts):
            mutated = mutate_fn(parent, rng)
            mutated = canonicalize(mutated) if mutated is not None else None
            if mutated is not None and mutated not in seen:
                child = mutated
                break
        if child is None:
            stalled += 1
            continue
        stalled = 0
        n_eval += 1
        value = _score(score_fn, child)
        seen[child] = Proposal(child, -np.inf if value is None else value, 1, parent)
        parents.append(child)
    ranked = _rank(p for p in seen.values() if np.isfinite(p.score))
    return SearchResult(ranked, n_eval, [ranked[0].score])


def compare_to_random(
    evolved: SearchResult, baseline: SearchResult, k: int = 10
) -> dict[str, float]:
    """Summarise evolved vs random-search proposals (novel molecules only, top-k)."""
    def stats(result: SearchResult) -> tuple[float, float]:
        top = result.top(k)
        if not top:
            return float("nan"), float("nan")
        return top[0].score, float(np.mean([p.score for p in top]))

    e_best, e_mean = stats(evolved)
    r_best, r_mean = stats(baseline)
    return {
        "evolved_best": e_best, "random_best": r_best,
        "evolved_top_k_mean": e_mean, "random_top_k_mean": r_mean,
        "evolved_evaluations": float(evolved.n_evaluations),
        "random_evaluations": float(baseline.n_evaluations),
        "improvement_top_k_mean": e_mean - r_mean,
    }


# ----------------------------------------------------------------------------- chemistry
_ELEMENTS = ("C", "N", "O", "F", "S", "Cl")
_ATOMIC_NUM = {"C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}
MIN_HEAVY, MAX_HEAVY = 5, 50

# Crude chemical-sanity filters for edits that RDKit accepts but that no one would make:
# peroxides, and neutral sulfur with 3-4 connections that has no S=O / S=N (sulfoxides,
# sulfones and sulfonamides are kept). This removes the worst nonsense; it does NOT check
# stability, synthesizability, toxicity or drug-likeness, so proposals still need expert review.
_IMPLAUSIBLE_SMARTS = ("[#8]-[#8]", "[SX3,SX4;!$([#16]=[#8]);!$([#16]=[#7]);!$([#16]=[#16])]")


def _random_atomic_number(rng: np.random.Generator) -> int:
    return _ATOMIC_NUM[_ELEMENTS[int(rng.integers(0, len(_ELEMENTS)))]]


def rdkit_canonicalize(smiles: str) -> str | None:
    """Canonical SMILES of the largest fragment; None if unparseable/out of bounds/implausible."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if not MIN_HEAVY <= mol.GetNumHeavyAtoms() <= MAX_HEAVY:
        return None
    for smarts in _IMPLAUSIBLE_SMARTS:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return None
    return Chem.MolToSmiles(mol)


def rdkit_mutate(smiles: str, rng: np.random.Generator) -> str | None:
    """Apply one random valid-by-construction edit; None if the edit does not sanitise.

    Edits: swap a non-aromatic heavy atom's element, append an atom to an atom with spare
    valence, delete a terminal atom, or change a non-aromatic bond's order.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    op = ("swap", "grow", "trim", "bond")[int(rng.integers(0, 4))]
    atoms = [a for a in rw.GetAtoms() if a.GetAtomicNum() > 1]
    if not atoms:
        return None
    try:
        if op == "swap":
            cand = [a for a in atoms if not a.GetIsAromatic()]
            if not cand:
                return None
            atom = cand[int(rng.integers(0, len(cand)))]
            atom.SetAtomicNum(_random_atomic_number(rng))
            atom.SetNoImplicit(False)
            atom.SetFormalCharge(0)
        elif op == "grow":
            cand = [a for a in atoms if not a.GetIsAromatic() and a.GetTotalNumHs() > 0]
            if not cand:
                return None
            atom = cand[int(rng.integers(0, len(cand)))]
            new = rw.AddAtom(Chem.Atom(_random_atomic_number(rng)))
            rw.AddBond(atom.GetIdx(), new, Chem.BondType.SINGLE)
        elif op == "trim":
            cand = [a for a in atoms if a.GetDegree() == 1]
            if not cand:
                return None
            rw.RemoveAtom(cand[int(rng.integers(0, len(cand)))].GetIdx())
        else:
            cand = [b for b in rw.GetBonds() if not b.GetIsAromatic() and not b.IsInRing()]
            if not cand:
                return None
            bond = cand[int(rng.integers(0, len(cand)))]
            order = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
            bond.SetBondType(order[int(rng.integers(0, 3))])
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return rdkit_canonicalize(Chem.MolToSmiles(out))
    except Exception:  # RDKit raises on valence/kekulisation errors; those edits are invalid
        return None
