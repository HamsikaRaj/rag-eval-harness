"""Tests for the DatasetManager service."""

import csv
import json
import pytest
from pathlib import Path

from services.dataset_manager.manager import DatasetManager, DatasetRecord, _compute_hash


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RECORDS = [
    {"question": "What is Paris?", "relevant_doc_ids": ["d1"], "ground_truth_answer": "Capital of France."},
    {"question": "What is Berlin?", "relevant_doc_ids": ["d2"], "ground_truth_answer": "Capital of Germany."},
    {"question": "What is Tokyo?",  "relevant_doc_ids": ["d3"], "ground_truth_answer": "Capital of Japan."},
    {"question": "What is Rome?",   "relevant_doc_ids": ["d4"], "ground_truth_answer": "Capital of Italy."},
    {"question": "What is Madrid?", "relevant_doc_ids": ["d5"], "ground_truth_answer": "Capital of Spain."},
]


@pytest.fixture
def mgr(tmp_path):
    return DatasetManager(data_dir=str(tmp_path))


@pytest.fixture
def json_file(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(SAMPLE_RECORDS))
    return path


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "samples.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "relevant_doc_ids", "ground_truth_answer"])
        writer.writeheader()
        for r in SAMPLE_RECORDS:
            writer.writerow({
                "question": r["question"],
                "relevant_doc_ids": ",".join(r["relevant_doc_ids"]),
                "ground_truth_answer": r["ground_truth_answer"],
            })
    return path


# ── Loading ───────────────────────────────────────────────────────────────────

def test_load_from_json(mgr, json_file):
    dataset_id = mgr.load_from_json(json_file, name="test")
    assert isinstance(dataset_id, str)
    assert len(dataset_id) > 0


def test_load_from_csv(mgr, csv_file):
    dataset_id = mgr.load_from_csv(csv_file, name="csv_test")
    record = mgr.get(dataset_id)
    assert record.sample_count == len(SAMPLE_RECORDS)
    assert record.source == "csv"
    # Verify CSV relevant_doc_ids were properly parsed
    assert isinstance(record.samples[0]["relevant_doc_ids"], list)


def test_load_inline(mgr):
    dataset_id = mgr.load_inline(SAMPLE_RECORDS, name="inline_test")
    record = mgr.get(dataset_id)
    assert record.name == "inline_test"
    assert record.sample_count == len(SAMPLE_RECORDS)


def test_load_with_documents(mgr):
    docs = [{"id": "d1", "text": "Paris info"}, {"id": "d2", "text": "Berlin info"}]
    dataset_id = mgr.load_inline(SAMPLE_RECORDS[:2], name="with_docs", documents=docs)
    record = mgr.get(dataset_id)
    assert len(record.documents) == 2


# ── Versioning ────────────────────────────────────────────────────────────────

def test_same_content_same_id(mgr):
    id1 = mgr.load_inline(SAMPLE_RECORDS, name="a")
    id2 = mgr.load_inline(SAMPLE_RECORDS, name="b")  # same content, different name
    assert id1 == id2  # hash is content-based


def test_different_content_different_id(mgr):
    id1 = mgr.load_inline(SAMPLE_RECORDS[:2], name="small")
    id2 = mgr.load_inline(SAMPLE_RECORDS[:3], name="bigger")
    assert id1 != id2


def test_sha256_hash_deterministic():
    h1 = _compute_hash(SAMPLE_RECORDS)
    h2 = _compute_hash(SAMPLE_RECORDS)
    assert h1 == h2


# ── Get + list ────────────────────────────────────────────────────────────────

def test_get_unknown_raises(mgr):
    with pytest.raises(KeyError):
        mgr.get("nonexistent")


def test_list_datasets(mgr):
    mgr.load_inline(SAMPLE_RECORDS[:2], name="d1")
    mgr.load_inline(SAMPLE_RECORDS[2:], name="d2")
    listing = mgr.list_datasets()
    assert len(listing) == 2
    assert all("dataset_id" in d for d in listing)


# ── Train / eval split ────────────────────────────────────────────────────────

def test_train_eval_split_ratio(mgr):
    dataset_id = mgr.load_inline(SAMPLE_RECORDS, name="split_test")
    train, eval_ = mgr.train_eval_split(dataset_id, train_ratio=0.8)
    total = len(train) + len(eval_)
    assert total == len(SAMPLE_RECORDS)
    assert len(train) >= 1 and len(eval_) >= 1


def test_split_is_deterministic(mgr):
    dataset_id = mgr.load_inline(SAMPLE_RECORDS, name="det_test")
    train1, eval1 = mgr.train_eval_split(dataset_id, seed=42)
    train2, eval2 = mgr.train_eval_split(dataset_id, seed=42)
    assert [s["question"] for s in train1] == [s["question"] for s in train2]


def test_different_seed_different_split(mgr):
    dataset_id = mgr.load_inline(SAMPLE_RECORDS, name="seed_test")
    train1, _ = mgr.train_eval_split(dataset_id, seed=1)
    train2, _ = mgr.train_eval_split(dataset_id, seed=99)
    # Very likely to differ with 5 samples and different seeds
    assert [s["question"] for s in train1] != [s["question"] for s in train2]


# ── Result storage ────────────────────────────────────────────────────────────

def test_store_and_retrieve_result(mgr):
    dataset_id = mgr.load_inline(SAMPLE_RECORDS[:2], name="result_test")
    result_data = {"run_id": "r1", "score": 0.85}
    mgr.store_result(dataset_id, "r1", result_data)
    loaded = mgr.get_result(dataset_id, "r1")
    assert loaded["score"] == 0.85


def test_get_result_missing_returns_none(mgr):
    dataset_id = mgr.load_inline(SAMPLE_RECORDS[:1])
    assert mgr.get_result(dataset_id, "nonexistent") is None


def test_list_results(mgr):
    dataset_id = mgr.load_inline(SAMPLE_RECORDS[:2])
    mgr.store_result(dataset_id, "r1", {})
    mgr.store_result(dataset_id, "r2", {})
    run_ids = mgr.list_results(dataset_id)
    assert set(run_ids) == {"r1", "r2"}
