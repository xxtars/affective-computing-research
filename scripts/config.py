from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RESEARCHERS_PATH = DATA_DIR / "researchers.json"
RESEARCHERS_ENRICHED_PATH = DATA_DIR / "researchers_enriched.json"
PAPERS_PATH = DATA_DIR / "papers.json"
PIPELINE_STATE_PATH = DATA_DIR / "pipeline_state.json"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
CLASSIFIED_PATH = DATA_DIR / "classified.json"
NEWLY_REJECTED_PATH = DATA_DIR / "newly_rejected_ids.json"

EMOTION_KEYWORDS = [
    "emotion", "affect", "affective", "sentiment",
    "facial expression", "micro-expression", "speech emotion",
    "mood", "empathy", "valence", "arousal",
    "emotion recognition", "multimodal emotion", "emotional",
]

DIRECTIONS = [
    "Affect Recognition",
    "Speech Emotion Recognition",
    "Multimodal Emotion Analysis",
    "Micro-expression",
    "Emotion-aware LLM",
    "Affective Brain-Computer Interface",
]

TAGS = [
    "survey", "benchmark", "dataset", "multimodal", "speech",
    "vision", "llm", "dialogue", "hci", "clinical",
    "cross-cultural", "foundation-model", "micro-expression",
]

S2_RPS = 1.0
CLAUDE_BATCH_SIZE = 15
