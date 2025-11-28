"""
Russell's Circumplex Model: Maps 28 GoEmotions to Valence/Arousal dimensions.
Based on Russell (1980) and VAD lexicons.
"""

import numpy as np
from typing import Dict


EMOTION_MAPPING = {
    'admiration': {'valence': 0.70, 'arousal': 0.30, 'dominance': -0.25},
    'amusement': {'valence': 0.75, 'arousal': 0.55, 'dominance': 0.40},
    'anger': {'valence': -0.80, 'arousal': 0.75, 'dominance': 0.55},
    'annoyance': {'valence': -0.60, 'arousal': 0.50, 'dominance': 0.20},
    'approval': {'valence': 0.55, 'arousal': 0.10, 'dominance': 0.25},
    'caring': {'valence': 0.70, 'arousal': 0.10, 'dominance': 0.20},
    'confusion': {'valence': -0.25, 'arousal': 0.35, 'dominance': -0.40},
    'curiosity': {'valence': 0.30, 'arousal': 0.45, 'dominance': 0.05},
    'desire': {'valence': 0.50, 'arousal': 0.55, 'dominance': 0.00},
    'disappointment': {'valence': -0.65, 'arousal': -0.20, 'dominance': -0.35},
    'disapproval': {'valence': -0.50, 'arousal': 0.05, 'dominance': 0.15},
    'disgust': {'valence': -0.80, 'arousal': 0.50, 'dominance': 0.30},
    'embarrassment': {'valence': -0.60, 'arousal': 0.55, 'dominance': -0.70},
    'excitement': {'valence': 0.80, 'arousal': 0.80, 'dominance': 0.60},
    'fear': {'valence': -0.70, 'arousal': 0.70, 'dominance': -0.60},
    'gratitude': {'valence': 0.80, 'arousal': 0.20, 'dominance': -0.10},
    'grief': {'valence': -0.90, 'arousal': -0.20, 'dominance': -0.60},
    'joy': {'valence': 0.85, 'arousal': 0.60, 'dominance': 0.50},
    'love': {'valence': 0.90, 'arousal': 0.50, 'dominance': 0.10},
    'nervousness': {'valence': -0.50, 'arousal': 0.60, 'dominance': -0.50},
    'optimism': {'valence': 0.65, 'arousal': 0.35, 'dominance': 0.45},
    'pride': {'valence': 0.70, 'arousal': 0.45, 'dominance': 0.65},
    'realization': {'valence': 0.20, 'arousal': 0.35, 'dominance': 0.10},
    'relief': {'valence': 0.65, 'arousal': -0.25, 'dominance': 0.30},
    'remorse': {'valence': -0.70, 'arousal': 0.10, 'dominance': -0.40},
    'sadness': {'valence': -0.70, 'arousal': -0.30, 'dominance': -0.50},
    'surprise': {'valence': 0.00, 'arousal': 0.70, 'dominance': -0.30},
    'neutral': {'valence': 0.00, 'arousal': 0.00, 'dominance': 0.00},
}


def calculate_vad(emotion_probs: Dict[str, float],
                  use_dominance: bool = False,
                  method: str = 'dominant') -> Dict[str, float]:
    """
    Calculate VAD scores from emotion probabilities.

    Args:
        emotion_probs: {emotion: probability} dict
        use_dominance: Include dominance dimension
        method: 'dominant' (use highest emotion) or 'weighted' (average all)

    Returns:
        Dict with 'valence', 'arousal', and optionally 'dominance'
    """
    if not emotion_probs or sum(emotion_probs.values()) == 0:
        result = {'valence': 0.0, 'arousal': 0.0}
        if use_dominance:
            result['dominance'] = 0.0
        return result

    if method == 'dominant':
        # Use VAD values from highest-probability emotion
        dominant_emo = max(emotion_probs.items(), key=lambda x: x[1])[0]

        if dominant_emo not in EMOTION_MAPPING:
            result = {'valence': 0.0, 'arousal': 0.0}
            if use_dominance:
                result['dominance'] = 0.0
            return result

        result = {
            'valence': float(EMOTION_MAPPING[dominant_emo]['valence']),
            'arousal': float(EMOTION_MAPPING[dominant_emo]['arousal'])
        }
        if use_dominance:
            result['dominance'] = float(EMOTION_MAPPING[dominant_emo]['dominance'])

        return result

    else:  # weighted
        # Average across all emotions (prone to cancellation bias)
        total = sum(emotion_probs.values())

        valence = sum(
            (prob / total) * EMOTION_MAPPING[emo]['valence']
            for emo, prob in emotion_probs.items()
            if emo in EMOTION_MAPPING
        )

        arousal = sum(
            (prob / total) * EMOTION_MAPPING[emo]['arousal']
            for emo, prob in emotion_probs.items()
            if emo in EMOTION_MAPPING
        )

        result = {'valence': float(valence), 'arousal': float(arousal)}

        if use_dominance:
            dominance = sum(
                (prob / total) * EMOTION_MAPPING[emo]['dominance']
                for emo, prob in emotion_probs.items()
                if emo in EMOTION_MAPPING
            )
            result['dominance'] = float(dominance)

        return result


def get_quadrant(valence: float, arousal: float) -> str:
    """Return circumplex quadrant based on valence/arousal."""
    if valence >= 0 and arousal >= 0:
        return 'Q1-Excited'
    elif valence < 0 and arousal >= 0:
        return 'Q2-Distressed'
    elif valence < 0 and arousal < 0:
        return 'Q3-Depressed'
    else:
        return 'Q4-Relaxed'


def get_intensity(valence: float, arousal: float) -> float:
    """Return Euclidean distance from origin."""
    return float(np.sqrt(valence**2 + arousal**2))


if __name__ == "__main__":
    # Validate mappings
    assert len(EMOTION_MAPPING) == 28, f"Expected 28 emotions, got {len(EMOTION_MAPPING)}"

    for emotion, values in EMOTION_MAPPING.items():
        assert -1 <= values['valence'] <= 1, f"{emotion} valence out of range"
        assert -1 <= values['arousal'] <= 1, f"{emotion} arousal out of range"
        assert -1 <= values['dominance'] <= 1, f"{emotion} dominance out of range"

    print("✓ All 28 emotions validated")

    # Test both methods
    test_probs = {'nervousness': 0.65, 'sadness': 0.45, 'fear': 0.30}

    print(f"\nTest probabilities: {test_probs}")

    scores_dominant = calculate_vad(test_probs, method='dominant')
    print(f"\nDominant method:")
    print(f"  Valence: {scores_dominant['valence']:.3f}")
    print(f"  Arousal: {scores_dominant['arousal']:.3f}")
    print(f"  Quadrant: {get_quadrant(scores_dominant['valence'], scores_dominant['arousal'])}")

    scores_weighted = calculate_vad(test_probs, method='weighted')
    print(f"\nWeighted method:")
    print(f"  Valence: {scores_weighted['valence']:.3f}")
    print(f"  Arousal: {scores_weighted['arousal']:.3f}")
    print(f"  Quadrant: {get_quadrant(scores_weighted['valence'], scores_weighted['arousal'])}")
