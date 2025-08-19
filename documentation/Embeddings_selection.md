Word2Vec (Google News 300D)

It’s one of the most widely used, pre-trained embeddings with very broad coverage of English vocabulary.

The Google News training corpus is massive and general enough to cover the kinds of words and expressions often found in lyrics.

It’s lightweight and fast to compute, making it efficient for large-scale SSM generation without heavy GPU requirements.

Chosen over GloVe or FastText because it strikes a good balance: high-quality semantic relationships, wide adoption, and simpler integration.

all-MiniLM-L6-v2 (384D)

It’s the most popular compact Sentence-Transformer, optimized for semantic similarity tasks.

Provides strong contextual embeddings at a fraction of the compute cost of BERT or RoBERTa.

Chosen because it’s small (6 layers, 384D), very efficient, and still highly ranked on semantic similarity benchmarks.

It generalizes well across domains, which is important since lyrics can vary widely in topic and style.

Avoided larger contextual models (like RoBERTa-large, GPT-style embeddings) because they’re too expensive for our pipeline without much extra gain.