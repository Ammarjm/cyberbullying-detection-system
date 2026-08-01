# Cyberbullying Detection Web App

This web interface was added to the existing cyberbullying detection project.

## What the page includes

- Text input box for a tweet or comment.
- Analyze button.
- Result section showing:
  - Cyberbullying type: `age`, `gender`, `religion`, `ethnicity`, `other_cyberbullying`, or `not_cyberbullying`.
  - Threat level from the system: `NOT_CB`, `LOW`, `HIGH`, or `CRITICAL`.
  - Severity score from `0.0` to `1.0`.
  - BERT toxicity score.
  - Rule score.
  - Context mitigation score.
- Confusion matrix image from `confusion_matrix.png`.
- Model performance summary from `evaluation_report.txt`.

## How to run

Open the project folder in Terminal or PowerShell, then run:

```bash
python app.py
```

Then open this link in the browser:

```text
http://127.0.0.1:5000
```

## If libraries are missing

Install the project requirements:

```bash
pip install -r requirements_web.txt
```

## Important files added

```text
app.py
requirements_web.txt
README_WEB_APP.md
templates/index.html
static/style.css
static/script.js
static/confusion_matrix.png
```

## Notes

The app uses the existing fine-tuned model folder:

```text
cyberbullying_bert_model
```

The app also reads the existing report file:

```text
evaluation_report.txt
```

If the BERT model cannot be loaded because a required library is missing, the original system code may fall back to its built-in heuristic mode.


## UX fixes added

- Threat badge colors now change by level: `NOT_CB`, `LOW`, `HIGH`, and `CRITICAL`.
- The model starts loading in the background when the server starts.
- The page shows a clear loading message while the model is warming up.
- The Analyze button shows an active loading state during analysis.
- Ctrl + Enter now runs the analysis from the text box.
