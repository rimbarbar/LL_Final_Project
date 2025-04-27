# CUISINARY: Audio Command Classification for a Cooking Assistant

## Overview
This project develops a voice-activated cooking assistant to classify 7 audio commands (`startrecipe`, `nextstep`, `repeatstep`, `timer`, `substitute`, `scale`, `done`) using the `facebook/wav2vec2-base` model. The dataset contains 70 audio clips (10 per class), recorded as `.m4a` and converted to WAV.

## Dataset
- **Source**: 70 audio clips in `/content/LL_final_project/`, organized into `/content/data/<command>/`.
- **EDA**:
  - 10 clips per class, balanced distribution.
  - Durations: ~1.2–4.5 seconds (mean ~2.5s).
  - Sample rate: 16kHz.
  - Visualizations: Duration histogram and sample waveforms show variability, with potential overlap (e.g., `timer` vs. `nextstep`).

## Methodology
- **Preprocessing**: Converted `.m4a` to WAV, used `Wav2Vec2Processor` for feature extraction.
- **Model**: Fine-tuned `facebook/wav2vec2-base` with 5-fold cross-validation (`StratifiedKFold`).
- **Training**:
  - 20 epochs, learning rate 5e-5, batch size 4, cosine scheduler, label smoothing (0.1).
  - Metrics: Accuracy, F1, precision, recall.
- **Challenges**: Small dataset (56 train/14 validation per fold), precision warnings due to missing predictions for some classes.

## Results
- **Average Accuracy**: 41.79% across 5 folds.
- **Fold Performance**:
  - Fold 1: 14.29% (no learning).
  - Fold 2: 78.57% (best).
  - Fold 3: 42.86%.
  - Fold 4: 57.14%.
  - Fold 5: 14.29% (no learning).
- **Visualizations**:
  - Accuracy/F1 per epoch: Fold 2 peaks at 78.57%, others stagnate.
  - Final accuracy bar plot: High variance (14.29%–78.57%).
- **Insights**: Small dataset and poor validation splits limit performance. Fold 2’s success suggests potential with more data or stronger augmentations.

## Setup
1. Clone the repo
2. Install dependencies: `pip install transformers datasets audiomentations pydub seaborn matplotlib numpy scikit-learn`
3. Run the notebook: `LL_Final_Project (2).ipynb` in Google Colab.
4. Data: Place `.m4a` files in `/content/LL_final_project/`.

## Future Work
- Collect more data (100+) to improve generalization.
- Add stronger augmentations (e.g., noise, pitch shift).
- Test lower learning rates (e.g., 1e-5) or more epochs.
- Deploy as a Streamlit app for real-time command recognition.

## Deliverables
- Notebook: `LL_Final_Project (2).ipynb`
- Visualizations: `audio_metadata.csv`, `duration_distribution.png`, `sample_waveforms.png`, `accuracy_per_fold.png`, `f1_per_fold.png`, `final_accuracy_bar.png`
- Project report: `LL Final Project Report_ CUISINARY`
