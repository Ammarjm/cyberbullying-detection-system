from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import json
import mimetypes
import os
import re
import threading
from typing import Any, Dict

from cyberbullying_detection_system import (
    BERTToxicityAnalyzer,
    IntegratedCyberbullyingDetectionSystem,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("CB_MODEL_DIR", BASE_DIR / "cyberbullying_bert_model"))
REPORT_PATH = BASE_DIR / "evaluation_report.txt"
INDEX_PATH = BASE_DIR / "templates" / "index.html"
STATIC_DIR = BASE_DIR / "static"

_system = None
_system_lock = threading.Lock()
_loading_started = False
_loading_error = None


def load_system() -> IntegratedCyberbullyingDetectionSystem:
    """Load the model once, then reuse it for every request."""
    global _system
    if _system is None:
        with _system_lock:
            if _system is None:
                custom_bert = BERTToxicityAnalyzer(model_name=str(MODEL_DIR))
                _system = IntegratedCyberbullyingDetectionSystem(bert_analyzer=custom_bert)
    return _system


def model_status() -> Dict[str, Any]:
    """Return a small status object for the front end."""
    loaded = _system is not None
    return {
        "loaded": loaded,
        "loading": bool(_loading_started and not loaded and _loading_error is None),
        "error": _loading_error,
        "model_dir_exists": MODEL_DIR.exists(),
    }


def start_model_warmup() -> None:
    """Start loading the model in the background so the first analysis is faster."""
    global _loading_started, _loading_error

    if _system is not None:
        return

    if _loading_started and _loading_error is None:
        return

    _loading_started = True
    _loading_error = None

    def warmup() -> None:
        global _loading_error
        try:
            load_system()
            print("AI model loaded and ready.")
        except Exception as exc:
            _loading_error = str(exc)
            print(f"Model warm-up failed: {exc}")

    thread = threading.Thread(target=warmup, daemon=True)
    thread.start()


def parse_evaluation_report() -> Dict[str, Any]:
    """Extract the main numbers from evaluation_report.txt for the home page."""
    report = {
        "test_set": "Not available",
        "accuracy": "Not available",
        "macro_precision": "Not available",
        "macro_recall": "Not available",
        "macro_f1": "Not available",
        "weighted_f1": "Not available",
        "grade": "Not available",
        "generated": "Not available",
        "raw_text": "evaluation_report.txt was not found.",
    }

    if not REPORT_PATH.exists():
        return report

    text = REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    report["raw_text"] = text

    patterns = {
        "generated": r"Generated:\s*(.+)",
        "test_set": r"Test set\s*:\s*(.+)",
        "accuracy": r"Accuracy\s*:\s*([0-9.]+\s*\([^)]+\))",
        "macro_precision": r"Macro Precision\s*:\s*([0-9.]+)",
        "macro_recall": r"Macro Recall\s*:\s*([0-9.]+)",
        "macro_f1": r"Macro F1-Score\s*:\s*([0-9.]+)",
        "weighted_f1": r"Weighted F1-Score\s*:\s*([0-9.]+)",
        "grade": r"Overall Grade\s*:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            report[key] = match.group(1).strip()

    return report


def analyze_text(text: str) -> Dict[str, Any]:
    if not text or not isinstance(text, str):
        raise ValueError("Please enter a text before analysis.")

    result = load_system().analyze(text.strip())
    return {
        "input_text": result.text,
        "predicted_type": result.predicted_type,
        "threat_level": result.threat_level.value,
        "severity_score": round(float(result.severity_score), 4),
        "bert_toxicity_score": round(float(result.p_toxic), 4),
        "rule_score": round(float(result.r_rule_soft), 4),
        "context_mitigation": round(float(result.c_context), 4),
        "mitigation_triggered": bool(result.mitigation_triggered),
        "actions": result.actions,
    }


class WebAppHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        """Keep console output short."""
        return

    def _send_bytes(self, content: bytes, status: int = 200, content_type: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(content, status=status, content_type="application/json; charset=utf-8")

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "File not found."}, status=404)
            return

        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), content_type=mime_type)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            self._send_file(INDEX_PATH)
            return

        if path == "/api/report":
            self._send_json(parse_evaluation_report())
            return

        if path == "/api/model-status":
            self._send_json(model_status())
            return

        if path == "/health":
            health = {
                "status": "ok",
                "model_dir": str(MODEL_DIR),
                "report_exists": REPORT_PATH.exists(),
            }
            health.update(model_status())
            self._send_json(health)
            return

        if path.startswith("/static/"):
            relative = path.replace("/static/", "", 1)
            target = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() in target.parents or target == STATIC_DIR.resolve():
                self._send_file(target)
            else:
                self._send_json({"error": "Invalid static path."}, status=400)
            return

        self._send_json({"error": "Route not found."}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/warmup":
            start_model_warmup()
            self._send_json(model_status())
            return

        if parsed.path != "/api/analyze":
            self._send_json({"error": "Route not found."}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body or "{}")
            text = (payload.get("text") or "").strip()
            self._send_json(analyze_text(text))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 5000), WebAppHandler)
    print("Cyberbullying Detection Web App")
    print("Open: http://127.0.0.1:5000")
    print("Loading AI model in the background. The first load may take 20-30 seconds.")
    print("Press Ctrl+C to stop.")
    start_model_warmup()
    server.serve_forever()


if __name__ == "__main__":
    main()
