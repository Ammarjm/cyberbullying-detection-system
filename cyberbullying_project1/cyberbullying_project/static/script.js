const textInput = document.getElementById("textInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const statusBox = document.getElementById("status");
const emptyState = document.getElementById("emptyState");
const resultPanel = document.getElementById("resultPanel");

const threatBadge = document.getElementById("threatBadge");
const scoreText = document.getElementById("scoreText");
const typeText = document.getElementById("typeText");
const levelText = document.getElementById("levelText");
const severityText = document.getElementById("severityText");
const bertText = document.getElementById("bertText");
const ruleText = document.getElementById("ruleText");
const contextText = document.getElementById("contextText");

const metricAccuracy = document.getElementById("metricAccuracy");
const metricPrecision = document.getElementById("metricPrecision");
const metricRecall = document.getElementById("metricRecall");
const metricMacroF1 = document.getElementById("metricMacroF1");
const metricWeightedF1 = document.getElementById("metricWeightedF1");
const metricGrade = document.getElementById("metricGrade");
const reportNote = document.getElementById("reportNote");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(message, isLoading = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle("loading", isLoading);
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "-";
}

function normalizeLevel(level) {
  return String(level || "").trim().toUpperCase();
}

function badgeClassForLevel(level) {
  const normalized = normalizeLevel(level);
  const classMap = {
    NOT_CB: "not-cb",
    LOW: "low",
    HIGH: "high",
    CRITICAL: "critical",
  };
  return classMap[normalized] || "unknown";
}

function updateThreatBadge(level) {
  const normalized = normalizeLevel(level);
  threatBadge.textContent = normalized || "-";
  threatBadge.className = `badge ${badgeClassForLevel(normalized)}`;
}

function showResult(data) {
  emptyState.classList.add("hidden");
  resultPanel.classList.remove("hidden");

  updateThreatBadge(data.threat_level);
  scoreText.textContent = formatScore(data.severity_score);
  typeText.textContent = data.predicted_type || "-";
  levelText.textContent = normalizeLevel(data.threat_level) || "-";
  severityText.textContent = formatScore(data.severity_score);
  bertText.textContent = formatScore(data.bert_toxicity_score);
  ruleText.textContent = formatScore(data.rule_score);
  contextText.textContent = formatScore(data.context_mitigation);
}

async function analyzeText() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus("Please enter a text before analysis.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  setStatus("Analyzing text. If this is the first request, the model may need 20–30 seconds", true);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Analysis failed.");
    }

    showResult(data);
    setStatus("Analysis completed.");
  } catch (error) {
    setStatus(error.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
}

async function warmupModel() {
  analyzeBtn.disabled = true;
  setStatus("Loading AI model in the background. First load may take 20–30 seconds", true);

  try {
    await fetch("/api/warmup", { method: "POST" });

    for (let attempt = 0; attempt < 90; attempt += 1) {
      const response = await fetch("/api/model-status");
      const status = await response.json();

      if (status.loaded) {
        analyzeBtn.disabled = false;
        setStatus("AI model is ready. Press Analyze or Ctrl + Enter.");
        return;
      }

      if (status.error) {
        analyzeBtn.disabled = false;
        setStatus(`Model warm-up failed: ${status.error}. You can still try Analyze.`);
        return;
      }

      setStatus("Loading AI model in the background. First load may take 20–30 seconds", true);
      await sleep(1000);
    }

    analyzeBtn.disabled = false;
    setStatus("The model is still loading. Wait a little, then click Analyze.", true);
  } catch (error) {
    analyzeBtn.disabled = false;
    setStatus("Could not check model loading status. You can still click Analyze.");
  }
}

async function loadReport() {
  try {
    const response = await fetch("/api/report");
    const report = await response.json();

    metricAccuracy.textContent = report.accuracy || "-";
    metricPrecision.textContent = report.macro_precision || "-";
    metricRecall.textContent = report.macro_recall || "-";
    metricMacroF1.textContent = report.macro_f1 || "-";
    metricWeightedF1.textContent = report.weighted_f1 || "-";
    metricGrade.textContent = report.grade || "-";
    reportNote.textContent = `Generated: ${report.generated || "-"} | Test set: ${report.test_set || "-"}`;
  } catch (error) {
    reportNote.textContent = "Could not load evaluation_report.txt.";
  }
}

analyzeBtn.addEventListener("click", analyzeText);

textInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    if (!analyzeBtn.disabled) {
      analyzeText();
    }
  }
});

clearBtn.addEventListener("click", () => {
  textInput.value = "";
  setStatus("");
  emptyState.classList.remove("hidden");
  resultPanel.classList.add("hidden");
  updateThreatBadge("LOW");
});

loadReport();
warmupModel();
