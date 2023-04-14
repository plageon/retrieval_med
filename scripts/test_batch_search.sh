python -m pyserini.search.lucene \
  --index indexes/bladder_cancer_train \
  --topics data/bladder_cancer_test_queries.tsv \
  --output output/retrieval/test_search_res.txt \
  --language zh \
  --bm25