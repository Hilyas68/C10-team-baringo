# C10 – Team Baringo — RAG Document Retrieval for Agronomic Advice

Retrieval engine for a RAG system surfacing agricultural-extension advice for smallholder farmers in Sub-Saharan Africa. Built for the Kaggle competition **agricultural-extension-rag-smart-retrieval-for-farmers**.

## Dataset

The knowledge base (`documents.csv`) contains 695 short agricultural-extension factsheets covering crop diseases, pests, nutrient deficiencies, soil management, and climate adaptation, spanning 13 crops and 21 Sub-Saharan African countries. 8.3% (58/695) are `llm_grounded` — rewrites of real CC-BY-licensed publications from FAO, CGIAR, Plantwise, IITA, ICRISAT, AGRA, and National Extension Services, each with a traceable `source_url`. The remaining 91.7% are `synthetic`: generated to extend topic/regional coverage, attributed to an institution's style but not externally verifiable.

Alongside the corpus: 308 training queries and 200 test queries (natural-language farmer questions), plus `qrels_train.csv` — 4,194 graded relevance judgments (0–3) linking queries to documents:

- **3 — Perfect**: directly answers the question.
- **2 — Relevant**: a genuinely complementary facet of the same topic.
- **1 — Marginal**: same issue, different crop.
- **0 — Not relevant**, including deliberate *hard negatives* — lexically similar documents that answer the wrong intent (e.g. "prevention" retrieved for a "causes" question).

Auditing `documents.csv` found real gaps: Maize/Tomato/Rice/Cassava make up 54.5% of documents while Pearl millet (a Sahelian staple) is only 1.4%; Nigeria/Ghana/Tanzania/Kenya make up 58% across just 4 of ~48 Sub-Saharan African countries. The corpus is also English-only and text-only. Full detail: `docs/data_card.pdf`.

The competition CSVs are bundled directly in `data/` (~480 KB total).

## Training Pipeline

Built incrementally, each stage measured against the previous before moving on:

1. **TF-IDF baseline** (provided): `TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)` over `title + ". " + text`, ranked by cosine similarity.
2. **BM25**, implemented from scratch (the kernel has no internet access, so `rank_bm25` couldn't be installed): raw term counts via `CountVectorizer`, plus manual IDF, term-frequency saturation, and document-length normalization (`k1=1.5, b=0.75`). Still purely lexical — confirmed it still fails to distinguish intent (e.g. "manage" vs. "prevent").
3. **Dense retrieval**: a MiniLM bi-encoder, attached offline via a Kaggle Model input (no internet for `huggingface.co` downloads). Loaded via raw `transformers.AutoModel`/`AutoTokenizer` rather than `sentence-transformers`, due to a `Pooling` version mismatch; mean-pooling and L2-normalization done by hand. Ranked by dot product (== cosine similarity). This is where semantic intent-matching first appears — a large jump over both lexical methods.
4. **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`, also attached offline, reranks the top-20 dense-retrieval candidates per query by jointly encoding `(query, document)` pairs. This fully resolved the "manage vs. prevent" hard-negative failure seen in stages 1–3.
5. **Fine-tuning experiment** (not adopted): fine-tuned the bi-encoder with a margin ranking loss (`margin=0.2`, `lr=2e-5`, batch 32, 1 epoch) on triplets from `train_queries`/`qrels_train` (positives: relevance ≥ 2; hard negatives: relevance = 0), holding out 61 of 308 queries for validation. Result: nDCG@5 on held-out queries **dropped** from 0.737 to 0.718 — likely overfitting from full-parameter fine-tuning on a small dataset. Kept the pretrained bi-encoder instead.

**Final pipeline**: dense retrieval (top-20) → cross-encoder rerank (top-5).

## Evaluation

All stages are scored with **nDCG@5** against `qrels_train`:

```
DCG@5  = Σ rel(d_i) / log2(i + 1)     for i = 1..5
IDCG@5 = DCG@5 of the ideal ranking (relevances sorted descending)
nDCG@5 = DCG@5 / IDCG@5
```

Documents absent from the judged pool count as relevance 0.

| Method | Train nDCG@5 | Leaderboard nDCG@5 |
|---|---|---|
| TF-IDF (baseline) | 0.503 | 0.551 (organizer-reported) |
| BM25 | 0.522 | — |
| Dense (bi-encoder) | 0.709 | 0.734 |
| Dense + cross-encoder rerank | 0.778 | **0.836** |
| Fine-tuned bi-encoder (held-out val only) | 0.718 (vs. 0.737 pretrained) | not submitted |

Fine-tuning used a held-out 61-query validation split (not in the training triplets) specifically to catch overfitting rather than trust in-sample gains — it caught exactly that.

## Reproduction

`src/retrieval.py` + `scripts/run.py` run **locally**, no Kaggle account needed — data is bundled in `data/`, and the bi-encoder/cross-encoder are pulled directly from Hugging Face.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run.py eval      # nDCG@5 for every stage on train_queries/qrels_train
python scripts/run.py submit    # writes submission.csv (dense retrieval + rerank)
```

Verified locally: `eval` reproduces the table above exactly (0.5031 / 0.5228 / 0.7083 / 0.7780); `submit` writes 1000 rows and reproduces the same top-5 for query 1001 (`4,1,3,5,2`) as our Kaggle-scored submission. First run downloads ~90 MB of weights; `submit` takes ~35s on CPU.

`scripts/retrieval_pipeline.ipynb` is the original exploratory notebook, built stage-by-stage inside a **Kaggle Notebook** (no internet — models attached as Kaggle Model inputs: `shree0910/minilm-sentence-transformer`, `johnsonhk88/cross-encoderms-marco-minilm-l-6-v2`). `src/retrieval.py` is a locally-runnable rewrite of the same logic — same models, same results.

> macOS note: an SSL error downloading from Hugging Face is a known Homebrew-Python/macOS trust-store mismatch, not a code bug — `truststore` (in `requirements.txt`) routes around it.

## Appendix

**Team**: C10 – Team Baringo

**Contributors**: Hassan Liasu, Gloria Paucara Cerpa - Problem statement.

