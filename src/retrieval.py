"""Core retrieval logic: TF-IDF baseline, BM25, dense bi-encoder retrieval,
and cross-encoder reranking, plus the nDCG@5 evaluation harness.

This mirrors, stage for stage, the exploratory work in scripts/retrieval_pipeline.ipynb
(originally developed offline in a Kaggle kernel with models attached as Kaggle Model
inputs). Here the same dense/cross-encoder models are pulled directly from Hugging Face,
since local runs have internet access.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

BIENCODER_MODEL_ID = os.environ.get("BIENCODER_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
CROSSENCODER_MODEL_ID = os.environ.get("CROSSENCODER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SHORTLIST_K = 20  # how many dense-retrieval candidates the cross-encoder reranks


# ---------------------------------------------------------------------------
# Data loading + evaluation
# ---------------------------------------------------------------------------

def load_data(data_dir):
    documents = pd.read_csv(os.path.join(data_dir, "documents.csv"))
    train_queries = pd.read_csv(os.path.join(data_dir, "train_queries.csv"))
    qrels_train = pd.read_csv(os.path.join(data_dir, "qrels_train.csv"))
    test_queries = pd.read_csv(os.path.join(data_dir, "test_queries.csv"))
    return documents, train_queries, qrels_train, test_queries


def doc_text_series(documents):
    return documents["title"] + ". " + documents["text"]


def build_qrels_lookup(qrels_train):
    """query_id -> {document_id: relevance grade}. Docs absent here count as relevance 0."""
    return qrels_train.groupby("query_id")[["document_id", "relevance"]].apply(
        lambda df: dict(zip(df["document_id"], df["relevance"]))
    ).to_dict()


def ndcg_at_k(ranked_doc_ids, relevance_map, k=5):
    relevances = [relevance_map.get(doc_id, 0.0) for doc_id in ranked_doc_ids[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

    ideal_relevances = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg_at_k(rankings_by_query_id, qrels_lookup, k=5):
    scores = [
        ndcg_at_k(doc_ids, qrels_lookup.get(qid, {}), k=k)
        for qid, doc_ids in rankings_by_query_id.items()
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Stage 1: TF-IDF baseline
# ---------------------------------------------------------------------------

def tfidf_rankings(documents, queries, top_k=5):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    doc_mat = vec.fit_transform(doc_text_series(documents))
    q_mat = vec.transform(queries["query"])
    sims = cosine_similarity(q_mat, doc_mat)

    doc_ids_array = documents["document_id"].values
    rankings = {}
    for i, qid in enumerate(queries["query_id"]):
        top_idx = sims[i].argsort()[::-1][:top_k]
        rankings[qid] = doc_ids_array[top_idx].tolist()
    return rankings


# ---------------------------------------------------------------------------
# Stage 2: BM25 (implemented directly; no external bm25 dependency)
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
        doc_term = self.cv.fit_transform(doc_text_series(documents))
        self.doc_term_csc = doc_term.tocsc()
        self.doc_lengths = np.asarray(doc_term.sum(axis=1)).ravel()
        self.avgdl = self.doc_lengths.mean()
        self.n_docs = doc_term.shape[0]
        df = np.asarray((doc_term > 0).sum(axis=0)).ravel()
        self.idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self.k1, self.b = k1, b

    def scores_for_query(self, query_text):
        q_terms = self.cv.transform([query_text]).indices
        scores = np.zeros(self.n_docs)
        for t in q_terms:
            tf = np.asarray(self.doc_term_csc[:, t].todense()).ravel()
            denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lengths / self.avgdl)
            scores += self.idf[t] * (tf * (self.k1 + 1)) / np.where(denom == 0, 1, denom)
        return scores


def bm25_rankings(documents, queries, top_k=5):
    bm25 = BM25Index(documents)
    doc_ids_array = documents["document_id"].values
    rankings = {}
    for i, qid in enumerate(queries["query_id"]):
        scores = bm25.scores_for_query(queries.iloc[i]["query"])
        top_idx = np.argsort(scores)[::-1][:top_k]
        rankings[qid] = doc_ids_array[top_idx].tolist()
    return rankings


# ---------------------------------------------------------------------------
# Stage 3: Dense bi-encoder retrieval
# ---------------------------------------------------------------------------

class DenseEncoder:
    """Mean-pooled, L2-normalized sentence embeddings from a bi-encoder transformer.

    Loaded via raw transformers (AutoModel/AutoTokenizer) rather than the
    sentence-transformers wrapper, matching the offline Kaggle setup where a
    sentence-transformers version mismatch made the high-level wrapper fail.
    """

    def __init__(self, model_id=BIENCODER_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed(self, texts, batch_size=32, max_length=256):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            with torch.no_grad():
                output = self.model(**encoded)
            pooled = self._mean_pooling(output, encoded["attention_mask"])
            normed = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normed)
        return torch.cat(all_embeddings, dim=0).numpy()


def dense_rankings(documents, queries, encoder=None, top_k=5, return_shortlist_k=None):
    """Returns top_k rankings, and optionally a wider shortlist per query for reranking."""
    encoder = encoder or DenseEncoder()
    doc_ids_array = documents["document_id"].values
    doc_embeddings = encoder.embed(doc_text_series(documents).tolist())
    query_embeddings = encoder.embed(queries["query"].tolist())

    sims = query_embeddings @ doc_embeddings.T  # dot product == cosine sim (both normalized)

    rankings, shortlists = {}, {}
    k = return_shortlist_k or top_k
    for i, qid in enumerate(queries["query_id"]):
        top_idx = np.argsort(sims[i])[::-1][:k]
        doc_ids = doc_ids_array[top_idx].tolist()
        shortlists[qid] = doc_ids
        rankings[qid] = doc_ids[:top_k]

    if return_shortlist_k:
        return rankings, shortlists, doc_embeddings
    return rankings


# ---------------------------------------------------------------------------
# Stage 4: Cross-encoder reranking
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    def __init__(self, model_id=CROSSENCODER_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

    def rerank(self, query_text, candidate_texts, batch_size=16):
        scores = []
        for i in range(0, len(candidate_texts), batch_size):
            batch_docs = candidate_texts[i:i + batch_size]
            encoded = self.tokenizer(
                [query_text] * len(batch_docs), batch_docs,
                padding=True, truncation=True, max_length=256, return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**encoded).logits.squeeze(-1)
            scores.extend(logits.tolist())
        return scores


def dense_plus_rerank_rankings(documents, queries, dense_encoder=None, reranker=None,
                                shortlist_k=SHORTLIST_K, top_k=5):
    dense_encoder = dense_encoder or DenseEncoder()
    reranker = reranker or CrossEncoderReranker()

    _, shortlists, _ = dense_rankings(
        documents, queries, encoder=dense_encoder, top_k=top_k, return_shortlist_k=shortlist_k
    )
    doc_text_lookup = dict(zip(documents["document_id"], doc_text_series(documents)))

    rankings = {}
    for i, qid in enumerate(queries["query_id"]):
        query_text = queries.iloc[i]["query"]
        shortlist_doc_ids = shortlists[qid]
        candidate_texts = [doc_text_lookup[d] for d in shortlist_doc_ids]

        ce_scores = reranker.rerank(query_text, candidate_texts)
        order = np.argsort(ce_scores)[::-1][:top_k]
        rankings[qid] = [shortlist_doc_ids[j] for j in order]

    return rankings


# ---------------------------------------------------------------------------
# Submission file writer
# ---------------------------------------------------------------------------

def write_submission(rankings, out_path, expected_query_ids=None):
    rows = [
        {"QueryId": qid, "DocumentId": int(doc_id)}
        for qid, doc_ids in rankings.items()
        for doc_id in doc_ids
    ]
    submission = pd.DataFrame(rows)

    assert submission.groupby("QueryId").size().eq(5).all(), "every query must have exactly 5 rows"
    if expected_query_ids is not None:
        assert set(submission["QueryId"]) == set(expected_query_ids), "every test query must appear"

    submission.to_csv(out_path, index=False)
    return submission
