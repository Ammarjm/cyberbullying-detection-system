# Fixes applied

- Replaced hardcoded Windows paths with portable paths based on the project folder.
- Added environment-variable overrides: `CB_BASE_DIR`, `CB_DATA_DIR`, `CB_DATASET_PATH`, and `CB_MODEL_DIR`.
- Fixed the class order so `not_cyberbullying` and `other_cyberbullying` match the trained label map.
- Removed the unsupported `SEXUAL_EXPLICIT` enum value from the dataset class enum.
- Changed training to an 80/10/10 stratified train/validation/test split.
- Validation is now used for best-model checkpoint selection. The test set is kept for final evaluation only.
- Updated `evaluate_model.py` to use the same untouched test split as training.
- Made `BERTFineTuner.evaluate()` use batched inference instead of row-by-row inference.
- Made `BERTFineTuner.train()` use the validation loader when provided.
- Updated the printed academic year to 2026.
- Updated `test_model.py` and `fix_model_labels.py` to use portable model paths.


## Patch applied after review
- Fixed `EvalDataset.__getitem__` inside `BERTFineTuner.evaluate()` by capturing `self.MAX_LENGTH` as `_max_length` before the nested dataset class.
- This prevents `AttributeError: 'EvalDataset' object has no attribute 'MAX_LENGTH'` during batched evaluation.
