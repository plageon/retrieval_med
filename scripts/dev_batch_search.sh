python -m pyserini.search.lucene \
  --index indexes/bladder_cancer_train \
  --topics data/bladder_cancer_dev_queries.tsv \
  --output output/retrieval/dev_search_res.txt \
  --language zh \
  --bm25