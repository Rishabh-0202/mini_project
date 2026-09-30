# Evaluation report (pipeline)

Model: `llama3.2` · embeddings: `nomic-embed-text` · reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Retrieval (no LLM)

| metric | value |
|---|---|
| hit@5 | 0.969 |
| hit_in_llm_context | 1.0 |
| MRR | 0.914 |
| contamination_rate | 0.0 |
| questions_scored | 32 |

## Answers

| metric | value |
|---|---|
| answered | 34 |
| abstain_accuracy | 1.0 |
| key_fact_recall | 0.639 |
| citation_rate | 0.938 |
| median_seconds | 60.9 |

## RAGAS

| metric | value |
|---|---|
| faithfulness | 0.64 |
| llm_context_precision_with_reference | 0.789 |
| answer_relevancy | 0.803 |
| n | 32 |
| n_scored | {'faithfulness': 32, 'llm_context_precision_with_reference': 31, 'answer_relevancy': 32} |
| judge | groq:openai/gpt-oss-120b |

## Per question

| id | abstain ok | key facts found / expected | s |
|---|---|---|---|
| GJ01 | ✅ | 1/2 ['1kw'] | 97.0 |
| GJ02 | ✅ | 1/3 ['50%'] | 53.4 |
| GJ03 | ✅ | 0/0 [] | 49.4 |
| GJ04 | ✅ | 0/0 [] | 42.4 |
| GJ05 | ✅ | 0/0 [] | 46.0 |
| GJ06 | ✅ | 0/6 [] | 44.1 |
| GJ07 | ✅ | 1/1 ['65%'] | 40.6 |
| GJ08 | ✅ | 1/1 ['6kw'] | 69.6 |
| GJ09 | ✅ | 0/1 [] | 77.7 |
| MH01 | ✅ | 0/1 [] | 61.9 |
| MH02 | ✅ | 1/1 ['70%'] | 4795.0 |
| MH03 | ✅ | 0/2 [] | 79.5 |
| MH04 | ✅ | 1/1 ['10kw'] | 73.2 |
| MH05 | ✅ | 3/4 ['20kw', 'rs100', 'rs500'] | 75.4 |
| MH06 | ✅ | 1/1 ['1mw'] | 60.7 |
| MH07 | ✅ | 0/0 [] | 56.4 |
| RJ01 | ✅ | 1/2 ['1mw'] | 27.4 |
| RJ02 | ✅ | 1/1 ['100%'] | 61.0 |
| RJ03 | ✅ | 1/1 ['50%'] | 30.7 |
| RJ04 | ✅ | 1/1 ['5mw'] | 41.8 |
| RJ05 | ✅ | 0/0 [] | 44.0 |
| KA01 | ✅ | 1/4 ['rs4.50'] | 48.3 |
| KA02 | ✅ | 4/5 ['1kw', '2000kw', '25year', 'rs3.74'] | 65.1 |
| KA03 | ✅ | 0/0 [] | 43.5 |
| KA04 | ✅ | 1/1 ['19%'] | 47.5 |
| TN01 | ✅ | 0/1 [] | 134.9 |
| TN02 | ✅ | 2/2 ['150kw', '999kw'] | 64.6 |
| TN03 | ✅ | 0/0 [] | 78.5 |
| CT01 | ✅ | 1/1 ['100kw'] | 71.1 |
| CT02 | ✅ | 1/1 ['30%'] | 77.7 |
| MS01 | ✅ | 1/2 ['70%'] | 114.9 |
| MS02 | ✅ | 1/3 ['1mw'] | 72.1 |
| AB01 | ✅ | 0/0 [] | 50.3 |
| AB02 | ✅ | 0/0 [] | 0.0 |
