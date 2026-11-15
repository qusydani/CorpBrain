"""
Offline tests for the multi-hop test set: chunk-group sampling, single-hop
labelling / merging, and per-type aggregation. No API calls or vector DBs.
"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import (  # noqa: E402
    HOP_KINDS, _aggregate_by_type, label_single_hop, merge_multihop, sample_chunk_groups,
)


def _chunks():
    """Two docs; chunks on nearby pages get similar embeddings."""
    rng = np.random.default_rng(0)
    out = []
    for source in ("a.pdf", "b.pdf"):
        for page in range(1, 11):
            emb = np.array([math.cos(page / 3), math.sin(page / 3), 0.1]) + rng.normal(0, 0.01, 3)
            out.append({"text": f"{source} p{page}", "source": source, "page": page, "emb": emb})
    return out


def test_groups_have_right_size_and_constraints():
    chunks = _chunks()
    kinds = list(HOP_KINDS) * 3
    groups = sample_chunk_groups(chunks, kinds, random.Random(1))

    assert [k for k, _ in groups] == kinds
    anchors = [idxs[0] for _, idxs in groups]
    assert len(set(anchors)) == len(anchors), "anchors must not repeat"
    for kind, idxs in groups:
        members = [chunks[i] for i in idxs]
        assert len(idxs) == HOP_KINDS[kind]["group"]
        assert len({(c["source"], c["page"]) for c in members}) == len(members)
        sources = {c["source"] for c in members}
        if HOP_KINDS[kind]["cross_doc"]:
            assert len(sources) == len(members)
        else:
            assert len(sources) == 1


def test_sampling_is_deterministic_with_seed():
    chunks = _chunks()
    kinds = list(HOP_KINDS)
    assert sample_chunk_groups(chunks, kinds, random.Random(7)) == \
           sample_chunk_groups(chunks, kinds, random.Random(7))


def test_cross_document_falls_back_with_one_source():
    chunks = [c for c in _chunks() if c["source"] == "a.pdf"]
    groups = sample_chunk_groups(chunks, ["cross-document"], random.Random(0))
    assert len(groups) == 1 and len(groups[0][1]) == 2


def test_label_single_hop_adds_type_and_sources():
    q = label_single_hop({"question": "q", "ground_truth": "g", "source": "a.pdf", "page": 3})
    assert q["type"] == "single-hop"
    assert q["sources"] == [{"source": "a.pdf", "page": 3}]
    assert q["source"] == "a.pdf"  # legacy fields kept


def test_merge_is_idempotent():
    singles = [{"question": "s1", "ground_truth": "g", "source": "a.pdf", "page": 1}]
    multi = [{"question": "m1", "ground_truth": "g", "type": "multi-hop",
              "sources": [{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": 2}]}]
    once = merge_multihop(singles, multi)
    twice = merge_multihop(once, multi)
    assert [q["type"] for q in once] == ["single-hop", "multi-hop"]
    assert twice == once


def test_aggregate_by_type():
    rows = [
        {"type": "single-hop", "accuracy": 1, "faithfulness": 0.8},
        {"type": "single-hop", "accuracy": 0, "faithfulness": float("nan")},
        {"accuracy": 1, "faithfulness": 0.6},  # untyped → single-hop
        {"type": "multi-hop", "accuracy": 0, "faithfulness": 0.4},
        {"type": "multi-hop", "accuracy": None, "faithfulness": 0.2},
    ]
    bt = _aggregate_by_type(rows, ["accuracy", "faithfulness"])
    assert bt["single-hop"]["n"] == 3
    assert bt["single-hop"]["accuracy"] == round(2 / 3, 4)
    assert bt["single-hop"]["faithfulness"] == 0.7      # NaN excluded
    assert bt["multi-hop"]["n"] == 2
    assert bt["multi-hop"]["accuracy"] == 0.0          # None excluded
    assert bt["multi-hop"]["faithfulness"] == 0.3
    # faithfulness is the only weighted metric present (accuracy has weight 0)
    assert bt["multi-hop"]["overall_score"] == 0.3


def test_aggregate_by_type_omits_missing_types():
    bt = _aggregate_by_type([{"accuracy": 1}], ["accuracy"])
    assert list(bt) == ["single-hop"]
