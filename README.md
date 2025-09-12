# SEEEM-TM

SEGMENTAÇÃO E EXTRAÇÃO DE ESTRUTURAS EMOCIONAIS EM MÚSICAS

# Lyric Segmentation ML

## Overview

This project provides machine learning models for segmenting song lyrics into verses and choruses using BiLSTM and CNN architectures. It incorporates features like repetition detection, rhyme analysis, and boundary-aware loss for improved accuracy.

## Installation

Clone the repository and install requirements:

git clone [repository-url]
cd lyric-segmentation-ml
pip install -r requirements.txt

## Training

Training is configured via YAML files in configs/training/.

Example:
./train.sh

Specific config:
python train_with_config.py configs/training/all_features_boundary_aware_loss.yaml

For CNN:
python train_cnn_with_config.py configs/training/cnn_test.yaml

## Prediction

Prediction uses models from training_sessions/.

Example:
./predict.sh

Specific:
python predict_baseline.py --session training_sessions/[session_dir]

For CNN:
python predict_cnn.py --session training_sessions/[cnn_session_dir]

## Client

Client was built using Streamlit, it uses the best 3 models with different training configuration.

To run:
`streamlit run client/app.py`

## Configuration

YAML configs in configs/ define parameters.

Training configs specify features, model params, loss.

Prediction configs handle inference settings like calibration.

## Features and Models

- Models: BiLSTM, CNN
- Features: SSM-based for head, tail, string, phonetic, POS
- Advanced: Attention mechanisms, boundary-aware loss

See documentation/ for more details.

## License

MIT
