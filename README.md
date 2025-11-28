# Post-Level Emotion Network Analysis for Mental Health Crisis Detection

Research analyzing emotion co-occurrence networks in mental health Reddit posts using transformer models and network science.

## Overview

This project develops post-level emotion analysis methods for mental health crisis detection in online communities, removing privacy barriers from user-level longitudinal tracking.

- **Dataset:** 40,745 Reddit posts from 18 mental health subreddits (2016-2023)
- **Methods:** Mental-BERT (28 GoEmotions), BERTopic (19 topics), Gaussian Graphical Models, Russell's Circumplex Model
- **Key Findings:** 69 stable emotion co-occurrence patterns, network structure generalizes across all conditions

## Key Results

### Emotion Classification (Mental-BERT)
- F1-macro: 0.38, F1-micro: 0.45
- Best performing emotions: Gratitude (F1=0.84), Love (F1=0.70), Remorse (F1=0.63)
- Challenging emotions: Pride (F1=0.12), Grief (F1=0.17) due to class imbalance

### Topic Modeling (BERTopic)
- 19 thematic clusters from 36,177 posts
- Crisis topics: Nervousness (0.45) + Sadness (0.34)
- Social anxiety: Embarrassment (0.34-0.36) elevated alongside nervousness
- Recovery topics: Pride (0.27) + Relief (0.26) + persistent Nervousness (0.32)

### Emotion Networks (GGM)
- 69 stable edges (bootstrap stability ≥0.90)
- Strongest connection: Fear-Nervousness (ρ=0.75)
- Network hubs: Sadness, Nervousness (high centrality across 4 metrics)
- 6 emotion communities: Anxiety, Depression, Irritability, Positive Affect, Uncertainty, Social

### Circumplex Mapping
- Q2-Distressed (high arousal, negative valence): 61% of posts
- Q3-Depressed (low arousal, negative valence): 14% of posts
- Negative valence states account for 75% of mental health discourse

### Cross-Condition Generalization
- Fear-nervousness edge appears in all 18/18 subreddits
- Network density varies minimally (0.14-0.18)
- Supports transdiagnostic emotion mechanisms

## Repository Structure

### Notebooks

#### `00_part_a_pipeline/`
Data collection and preprocessing (Part A of project)
- **part_a_data_preparation_pipeline.ipynb** - Complete pipeline: Reddit API collection (18 subreddits × 6 methods), deduplication (109K → 40K posts), multi-track preprocessing, quality assessment

#### `01_data_preparation/`
GoEmotions dataset preparation
- **goemotions_dataset_loader.ipynb** - Loads Google Research GoEmotions (211K → 57K unique), creates train/val/test splits, generates class weights for imbalanced data

#### `02_model_training/`
Transformer model fine-tuning
- **mental_bert_goemotions_finetune.ipynb** - Mental-BERT fine-tuning with Focal Loss, Optuna hyperparameter optimization, per-emotion optimal thresholds (selected model)
- **mental_roberta_goemotions_finetune.ipynb** - Mental-RoBERTa alternative (benchmarking only)

#### `03_inference/`
Emotion classification on Reddit data
- **mental_health_reddit_inference.ipynb** - Apply Mental-BERT to 40K posts, chunk strategy for long posts (510 tokens, 50 overlap), batch inference with A100 GPU

#### `04_topic_modeling/`
Thematic clustering
- **mental_health_bertopic_analysis.ipynb** - BERTopic with sentence-transformers/all-mpnet-base-v2, HDBSCAN clustering, outlier reduction (33.9% → 12.8%), LLaMA topic labeling

#### `05_circumplex_mapping/`
Valence-Arousal emotion mapping
- **add_circumplex_features.ipynb** - Maps 28 GoEmotions to Russell's Circumplex Model, dominant emotion method
- **circumplex_validation.ipynb** - Validates V-A distributions, quadrant analysis

#### `06_network_analysis/`
Emotion co-occurrence networks
- **mental_health_emotion_network_analysis.ipynb** - Gaussian Graphical Model (GraphicalLassoCV), bootstrap validation (1,000 iterations), centrality analysis, community detection (Louvain), cross-subreddit generalization

#### `07_exploratory/`
Additional visualizations
- **comprehensive_emotion_analysis.ipynb** - Topic-emotion heatmaps, radar charts, z-score analysis, correlation matrices, hierarchical clustering

### Source Code

#### `src/`
- **emotion_circumplex_mapping.py** - Valence-Arousal mappings for all 28 GoEmotions

### Results

#### `results/topic_modeling/`
- **topic_info.csv** - 19 topics with keywords, post counts
- **summary_statistics.csv** - Top 3 emotions per topic with probabilities

#### `results/emotion_network/`
- **stable_edges.csv** - 69 stable edges with partial correlations, bootstrap stability
- **emotion_centrality.csv** - Centrality metrics (strength, betweenness, closeness, eigenvector) for 28 emotions
- **partial_correlations.csv** - Full partial correlation matrix
- **subreddit_network_comparison.csv** - Network statistics across 18 subreddits
- **topic_network_comparison.csv** - Network statistics across 19 topics

#### `results/subreddit_networks/`
Individual network files for all 18 subreddits (edges + partial correlations)

#### `results/model_metadata/`
- **optimal_thresholds.json** - Per-emotion classification thresholds (Mental-BERT)
- **metadata.json** - Training metrics, F1 scores, hyperparameters

#### `results/bertopic/` and `results/networks/`
- **bertopic/** - Interactive topic visualizations (HTML): intertopic distance map, topic barchart, hierarchy dendrogram
- **networks/** - Network visualizations (PNG + HTML): global network, 18 subreddit networks, 19 topic networks, comparison heatmaps

### Data

Large data files (182-236MB) hosted on Google Drive due to GitHub file size limits.

See [`data/README.md`](data/README.md) for download links.

## Installation & Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Python Environment
- Python 3.8+
- PyTorch 2.0+
- CUDA support recommended for model training/inference

### Hardware
Notebooks were executed on Google Colab with:
- NVIDIA A100 GPU (80GB VRAM)
- 167GB RAM

For inference/analysis only, CPU or smaller GPUs sufficient.

## Usage

### Running Notebooks

Notebooks are organized sequentially. Two starting points:

**Option 1: Start from scratch (Part A)**
1. Run `00_part_a_pipeline/part_a_data_preparation_pipeline.ipynb` - Requires Reddit API credentials
2. Continue with `01_data_preparation/goemotions_dataset_loader.ipynb`

**Option 2: Use preprocessed data (Part B)**
1. Download preprocessed data from Google Drive (see `data/README.md`)
2. Start from `02_model_training/` or later stages

### Key Dependencies
- **Mental-BERT model:** `mental/mental-bert-base-uncased` (HuggingFace)
- **BERTopic base model:** `sentence-transformers/all-mpnet-base-v2` (HuggingFace)
- **GoEmotions dataset:** [Google Research](https://github.com/google-research/google-research/tree/master/goemotions) or (HuggingFace)

## Project Background

This research addresses three gaps in computational mental health detection:

1. **Privacy:** Few research methods were tracking users over months; post-level analysis enables immediate assessment without longitudinal tracking
2. **Emotion networks:** Prior work uses individual emotion frequencies; this models co-occurrence patterns via partial correlations
3. **Circumplex validation:** Russell's model validated in labs; this tests applicability to naturalistic mental health text

## Subreddits Analyzed (18 total)

**Mood Disorders:** depression, bipolar, BipolarReddit
**Anxiety Disorders:** Anxiety, socialanxiety, OCD
**Neurodevelopmental:** ADHD, autism, aspergers
**Crisis Support:** SuicideWatch
**Recovery:** getting_over_it, decidingtobebetter, selfhelp
**Other:** BPD (personality), ptsd (trauma), eating_disorders, addiction, mentalhealth (general)

## Citation

If you use this code or data, please cite:

```
[Surya Embar] (2025). Post-Level Emotion Network Analysis for Mental Health Crisis Detection
in Online Communities.
```


## Acknowledgments

- Mental-BERT: Ji et al. (2022)
- GoEmotions: Demszky et al. (2020)
- Reddit data collected via PRAW API
- Analysis conducted on Google Colab A100 infrastructure
