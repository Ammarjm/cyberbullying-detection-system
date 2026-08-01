

import os
import shutil
from pathlib import Path
from transformers import AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
model_path = Path(os.getenv("CB_MODEL_DIR", BASE_DIR / "cyberbullying_bert_model"))
temp_path = model_path.with_name(model_path.name + "_temp")

print(f"Loading model from {model_path}...")
model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

id2label = {
    0: "age",
    1: "ethnicity",
    2: "gender",
    3: "religion",
    4: "not_cyberbullying",
    5: "other_cyberbullying"
}

label2id = {v: k for k, v in id2label.items()}

model.config.id2label = id2label
model.config.label2id = label2id

print(f"Saving to temporary directory: {temp_path}")
model.save_pretrained(str(temp_path))

if model_path.exists():
    print(f"Removing old model directory: {model_path}")
    shutil.rmtree(str(model_path))

print(f"Moving {temp_path} to {model_path}")
shutil.move(str(temp_path), str(model_path))

print("✅ Label mappings updated successfully!")
print("Now the model will return class names like 'age', 'religion', etc.")