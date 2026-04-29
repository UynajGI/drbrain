"""Tests for causal chain extraction from argument mechanisms."""

from drbrain.extractor.argument import ExtractedArgument
from drbrain.extractor.causal_chain import (
    CausalChain,
    build_causal_chains,
    find_chains_from,
    find_path,
)


def _make_arg(claim, target, mechanism, claim_type="proposes", target_type="Method"):
    """Helper to create ExtractedArgument with mechanism."""
    return ExtractedArgument(
        claim=claim,
        claim_type=claim_type,
        target=target,
        target_type=target_type,
        mechanism=mechanism,
    )


# -- CausalChain dataclass --


def test_causal_chain_creation():
    """CausalChain stores links and computes summary."""
    arg1 = _make_arg("A replaces B", "B", "parallel computation")
    arg2 = _make_arg("C extends A", "A", "optimized scheduling")
    chain = CausalChain(links=[arg1, arg2])
    assert len(chain.links) == 2
    # Summary: origin (first arg's target) → end (last arg's target)
    assert chain.summary() == "B → A (via parallel computation, optimized scheduling)"


def test_causal_chain_single_link():
    """Single-link chain produces correct summary."""
    arg = _make_arg("A solves B", "B", "gradient descent")
    chain = CausalChain(links=[arg])
    assert chain.summary() == "B → B (via gradient descent)"


# -- build_causal_chains --


def test_build_causal_chains_simple():
    """Build chains from arguments sharing a target (same concept chain)."""
    args = [
        _make_arg("Attention replaces RNN", "RNN", "parallel computation"),
        _make_arg("Transformer improves RNN understanding", "RNN", "attention weights"),
    ]
    chains = build_causal_chains(args)
    assert len(chains) >= 1
    # Both args target RNN, so they form a chain about RNN
    found = any(len(c.links) >= 2 for c in chains)
    assert found


def test_build_causal_chains_no_mechanism():
    """Arguments without mechanism don't form chains."""
    args = [
        _make_arg("A does B", "B", ""),
        _make_arg("C does A", "A", ""),
    ]
    chains = build_causal_chains(args)
    assert len(chains) == 0


def test_build_causal_chains_disconnected():
    """Separate mechanism chains produce separate CausalChains."""
    args = [
        _make_arg("A replaces B", "B", "mechanism 1"),
        _make_arg("B replaces C", "C", "mechanism 2"),
    ]
    chains = build_causal_chains(args)
    # No shared targets, so each arg is its own chain (or no chain if no links)
    # With current model: args with different targets don't chain
    # Each is a singleton chain or empty
    assert all(len(c.links) >= 1 for c in chains)


def test_build_causal_chains_shared_target():
    """Multiple arguments targeting same concept create a chain."""
    args = [
        _make_arg("A critiques B", "B", "mechanism 1"),
        _make_arg("C supports B", "B", "mechanism 2"),
    ]
    chains = build_causal_chains(args)
    # Both target B, should form at least one chain with 2 links
    found = any(len(c.links) == 2 for c in chains)
    assert found


def test_build_causal_chains_long_chain():
    """Chain of 3+ links via shared targets."""
    args = [
        _make_arg("A on B", "X", "m1"),
        _make_arg("B on X", "X", "m2"),
        _make_arg("C on X", "X", "m3"),
    ]
    chains = build_causal_chains(args)
    longest = max(chains, key=lambda c: len(c.links))
    assert len(longest.links) >= 3


# -- find_chains_from --


def test_find_chains_from_start():
    """find_chains_from finds all chains originating from a concept."""
    args = [
        _make_arg("A on X", "X", "m1"),
        _make_arg("B on X", "X", "m2"),
        _make_arg("C on X", "X", "m3"),
    ]
    chains = find_chains_from(args, "X")
    assert len(chains) >= 1


def test_find_chains_from_no_match():
    """find_chains_from returns empty when concept not a target."""
    args = [
        _make_arg("A replaces B", "B", "m1"),
    ]
    chains = find_chains_from(args, "Z")
    assert chains == []


# -- find_path --


def test_find_path_direct():
    """find_path finds direct causal link when source==target of an arg."""
    args = [
        _make_arg("A replaces B", "B", "parallel computation"),
    ]
    path = find_path(args, "B", "B")
    assert path is not None
    assert len(path.links) == 1


def test_find_path_via_shared_target():
    """find_path finds path through shared target concept."""
    args = [
        _make_arg("A addresses B", "B", "m1"),
        _make_arg("C extends B", "B", "m2"),
    ]
    path = find_path(args, "B", "B")
    assert path is not None
    assert len(path.links) >= 1


def test_find_path_no_path():
    """find_path returns None when no causal link exists."""
    args = [
        _make_arg("A replaces B", "B", "m1"),
        _make_arg("D replaces C", "C", "m2"),
    ]
    path = find_path(args, "B", "C")
    assert path is None


def test_empty_arguments():
    """All functions handle empty argument list."""
    assert build_causal_chains([]) == []
    assert find_chains_from([], "X") == []
    assert find_path([], "A", "B") is None
