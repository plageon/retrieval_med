python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input collection \
  --language zh \
  --index indexes/bladder_cancer_train \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw