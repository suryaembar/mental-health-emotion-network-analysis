# Data Files

Due to GitHub's file size limitations (100MB), large data files are hosted on Google Drive.

## Download Links

### Preprocessed Reddit Data
- **reddit_all_subreddits_preprocessed_20250821_175216.json** (182 MB)
  - 40,745 preprocessed posts from 18 mental health subreddits
  - Fields: text_raw, text_light, text_ml_bow, quality_tier, quality_flags, category, subreddit, title, selftext, created_utc, score, num_comments, text_length
  - Download: [Google Drive Link - To be added]

### Emotion Predictions
- **reddit_emotions_predicted_20251118_194626.json** (231 MB)
  - Mental-BERT emotion classification results
  - Fields: All preprocessed fields + emotion_probabilities (28 emotions), top_3_emotions, dominant_emotion, predicted_emotion_labels
  - Download: [Google Drive Link - To be added]

### Circumplex Mapping
- **reddit_with_circumplex_20251120_192036.json** (236 MB)
  - Russell's Circumplex Model (Valence × Arousal) mapping
  - Fields: All emotion prediction fields + valence, arousal, dominance, quadrant, intensity
  - Download: [Google Drive Link - To be added]

### Topic Modeling
- **reddit_with_topics_20251121_230630.json** (211 MB)
  - BERTopic thematic clustering results
  - Fields: All circumplex fields + topic, topic_probability, representative_document
  - Download: [Google Drive Link - To be added]

## File Placement

After downloading, place files in the following locations:

```
data/
├── reddit_all_subreddits_preprocessed_20250821_175216.json
├── reddit_emotions_predicted_20251118_194626.json
├── reddit_with_circumplex_20251120_192036.json
└── reddit_with_topics_20251121_230630.json
```

## Data Processing Pipeline

The data files represent sequential stages of the analysis pipeline:

1. **Preprocessed** → Mental-BERT classification → **Emotion Predictions**
2. **Emotion Predictions** → Circumplex mapping → **With Circumplex**
3. **With Circumplex** → BERTopic clustering → **With Topics**

Each subsequent file includes all fields from previous stages plus new features.

## Data Format

All files are JSON arrays of objects. Load with:

```python
import pandas as pd
df = pd.read_json('reddit_all_subreddits_preprocessed_20250821_175216.json')
```

## Dataset Statistics

- **Total posts**: 40,745
- **Subreddits**: 18 mental health communities
- **Date range**: 2016-2023
- **Quality distribution**: Tier 1 (90.58%), Tier 2 (6.32%), Tier 3 (3.10%)
- **Emotions classified**: 28 GoEmotions categories
- **Topics discovered**: 19 thematic clusters
- **Topic assignment rate**: 87.18% (31,539 posts)
