"""Modular feature extraction system."""

from .head_ssm import (
    HeadSSMExtractor,
    extract_head_ssm_features,
    compute_head_ssm,
    summarize_ssm_per_line
)
from .tail_ssm import (
    TailSSMExtractor,
    extract_tail_ssm_features,
    compute_tail_ssm,
    extract_tail
)
from .phonetic_ssm import (
    PhoneticSSMExtractor,
    extract_phonetic_ssm_features,
    compute_phonetic_ssm,
    PhoneticProcessor,
    get_phonetic_processor,
    initialize_phonetic_processor,
    compute_phoneme_similarity,
    compute_signature_similarity,
    normalize_features_per_song
)
from .pos_ssm import (
    POSSSMExtractor,
    extract_pos_ssm_features,
    compute_pos_ssm,
    extract_pos_sequence,
    compute_pos_similarity
)
from .string_ssm import (
    StringSSMExtractor,
    extract_string_ssm_features,
    compute_string_ssm,
    normalized_levenshtein_similarity,
    normalize_string
)
from .word2vec_embeddings import (
    Word2VecEmbeddingsExtractor,
    get_word2vec_model,
    text_to_word2vec_embedding,
    compute_word2vec_ssm,
    extract_word2vec_summary_features
)
from .contextual_embeddings import (
    ContextualEmbeddingsExtractor,
    get_sentence_transformer_model,
    text_to_contextual_embedding,
    compute_contextual_ssm,
    extract_contextual_summary_features
)
from .extractor import (
    FeatureExtractor,
    create_feature_extractor,
    validate_feature_config
)

__all__ = [
    'HeadSSMExtractor',
    'extract_head_ssm_features', 
    'compute_head_ssm',
    'summarize_ssm_per_line',
    'TailSSMExtractor',
    'extract_tail_ssm_features',
    'compute_tail_ssm',
    'extract_tail',
    'PhoneticSSMExtractor',
    'extract_phonetic_ssm_features',
    'compute_phonetic_ssm',
    'PhoneticProcessor',
    'get_phonetic_processor',
    'initialize_phonetic_processor',
    'compute_phoneme_similarity',
    'compute_signature_similarity',
    'normalize_features_per_song',
    'POSSSMExtractor',
    'extract_pos_ssm_features',
    'compute_pos_ssm',
    'extract_pos_sequence',
    'compute_pos_similarity',
    'StringSSMExtractor',
    'extract_string_ssm_features',
    'compute_string_ssm',
    'normalized_levenshtein_similarity',
    'normalize_string',
    'Word2VecEmbeddingsExtractor',
    'get_word2vec_model',
    'text_to_word2vec_embedding',
    'compute_word2vec_ssm',
    'extract_word2vec_summary_features',
    'ContextualEmbeddingsExtractor',
    'get_sentence_transformer_model',
    'text_to_contextual_embedding',
    'compute_contextual_ssm',
    'extract_contextual_summary_features',
    'FeatureExtractor',
    'create_feature_extractor',
    'validate_feature_config'
]
