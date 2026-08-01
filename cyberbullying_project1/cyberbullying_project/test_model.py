
import os
from pathlib import Path

from cyberbullying_detection_system import (
    IntegratedCyberbullyingDetectionSystem,
    BERTToxicityAnalyzer,
)

# Get the current project directory.
BASE_DIR = Path(__file__).resolve().parent

# Use the fine-tuned model folder inside the project.
# You can override this path by setting the CB_MODEL_DIR environment variable.
MODEL_DIR = Path(os.getenv("CB_MODEL_DIR", BASE_DIR / "cyberbullying_bert_model"))

# Load the fine-tuned BERT model.
custom_bert = BERTToxicityAnalyzer(model_name=str(MODEL_DIR))

# Create the integrated cyberbullying detection system.
system = IntegratedCyberbullyingDetectionSystem(bert_analyzer=custom_bert)

# Test examples.
examples = [
    "You are ugly.",
    "Women are too emotional to be leaders.",
    "I respect your opinion but disagree.",
    "All Muslims are terrorists.",
    "That's so gay.",
    "Old people don't understand technology.",
]

print("=" * 60)
print("Testing fine-tuned model predictions")
print("=" * 60)

for text in examples:
    # Analyze the text.
    result = system.analyze(text)

    # Print the result in English.
    print(f"Text: {text}")
    print(f"Predicted type: {result.predicted_type}")
    print(f"Threat level: {result.threat_level.value}")
    print(f"Severity score: {result.severity_score:.4f}")
    print(f"BERT toxicity score: {result.p_toxic:.4f}")
    print(f"Rule score: {result.r_rule_soft:.4f}")
    print(f"Context mitigation: {result.c_context:.4f}")
    print("-" * 60)