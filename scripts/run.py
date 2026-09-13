#!/usr/bin/env python3
"""CLI for the retrieval pipeline.

Usage:
    python scripts/run.py eval      # print nDCG@5 for every stage on train_queries/qrels_train
    python scripts/run.py submit    # write submission.csv from the best pipeline (dense + rerank)
"""

import argparse
import os
import sys
import time

import truststore

# Some systems (notably macOS + Homebrew Python) ship an OpenSSL trust store that
# rejects certs the OS itself accepts, breaking Hugging Face downloads. Route SSL
# verification through the OS trust store instead, before transformers opens any connection.
truststore.inject_into_ssl()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval import (  # noqa: E402
    BM25Index,
    CrossEncoderReranker,
    DenseEncoder,
    bm25_rankings,
    build_qrels_lookup,
    dense_plus_rerank_rankings,
    dense_rankings,
    load_data,
    mean_ndcg_at_k,
    tfidf_rankings,
    write_submission,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(REPO_ROOT, "data"))
SUBMISSION_PATH = os.environ.get("SUBMISSION_PATH", os.path.join(REPO_ROOT, "submission.csv"))


def cmd_eval(args):
    documents, train_queries, qrels_train, _ = load_data(DATA_DIR)
    qrels_lookup = build_qrels_lookup(qrels_train)

    t0 = time.time()
    tfidf_r = tfidf_rankings(documents, train_queries)
    print(f"TF-IDF            nDCG@5 = {mean_ndcg_at_k(tfidf_r, qrels_lookup):.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    bm25_r = bm25_rankings(documents, train_queries)
    print(f"BM25              nDCG@5 = {mean_ndcg_at_k(bm25_r, qrels_lookup):.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    encoder = DenseEncoder()
    dense_r = dense_rankings(documents, train_queries, encoder=encoder)
    print(f"Dense bi-encoder  nDCG@5 = {mean_ndcg_at_k(dense_r, qrels_lookup):.4f}  ({time.time()-t0:.1f}s)")

    if not args.skip_rerank:
        t0 = time.time()
        reranker = CrossEncoderReranker()
        rerank_r = dense_plus_rerank_rankings(documents, train_queries, dense_encoder=encoder, reranker=reranker)
        print(f"Dense + rerank    nDCG@5 = {mean_ndcg_at_k(rerank_r, qrels_lookup):.4f}  ({time.time()-t0:.1f}s)")


def cmd_submit(args):
    documents, _, _, test_queries = load_data(DATA_DIR)

    encoder = DenseEncoder()
    reranker = CrossEncoderReranker()
    rankings = dense_plus_rerank_rankings(documents, test_queries, dense_encoder=encoder, reranker=reranker)

    submission = write_submission(rankings, SUBMISSION_PATH, expected_query_ids=test_queries["query_id"])
    print(f"Wrote {SUBMISSION_PATH} ({submission.shape[0]} rows, {submission['QueryId'].nunique()} queries)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval", help="Score every retrieval stage on train_queries/qrels_train")
    eval_parser.add_argument("--skip-rerank", action="store_true", help="Skip the slower cross-encoder stage")
    eval_parser.set_defaults(func=cmd_eval)

    submit_parser = sub.add_parser("submit", help="Write submission.csv using dense retrieval + rerank")
    submit_parser.set_defaults(func=cmd_submit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
