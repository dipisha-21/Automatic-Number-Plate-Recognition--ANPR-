from flask import Flask, request, render_template_string
from pathlib import Path
import os
import tempfile
import threading
import urllib.request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_PATH = Path(os.getenv("ANPR_MODEL_PATH", "/tmp/anpr-best.pt"))
MODEL_URL = os.getenv(
    "ANPR_MODEL_URL",
    "https://github.com/dipisha-21/Automatic-Number-Plate-Recognition--ANPR-/releases/download/anpr-model-demo/best.pt",
)
_pipeline = None
_pipeline_lock = threading.Lock()

HTML = '''
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>ANPR Demo</title>
<style>body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;background:#f6f8fb;color:#20242b}.card{background:#fff;padding:24px;border-radius:14px;box-shadow:0 4px 16px #0001}.result{margin-top:20px;padding:16px;background:#f0f4ff;border-radius:10px}.note{margin-top:18px;color:#596273;font-size:14px;line-height:1.5}input,button{padding:10px;margin-top:10px}button{cursor:pointer}</style></head><body>
<h1>Automatic Number Plate Recognition</h1><p>YOLOv8 + EasyOCR portfolio demo.</p><div class="card">
<form method="post" enctype="multipart/form-data"><input type="file" name="image" accept="image/*" required><br><button type="submit">Run ANPR</button></form>
{% if message %}<div class="result">{{ message }}</div>{% endif %}
</div><p class="note">Upload a vehicle image to run plate detection and OCR. This is a portfolio prototype; model quality is limited by the small retraining dataset currently stored in this repository.</p></body></html>
'''


def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".download")
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    if not tmp_path.exists() or tmp_path.stat().st_size < 100000:
        raise RuntimeError("Downloaded model file is missing or unexpectedly small.")
    tmp_path.replace(MODEL_PATH)
    return MODEL_PATH


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from src.anpr.pipeline import ANPRPipeline
                _pipeline = ANPRPipeline(str(ensure_model()), gpu=False)
    return _pipeline


@app.route('/', methods=['GET', 'POST'])
def home():
    message = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            message = 'Please choose an image.'
        else:
            suffix = Path(file.filename).suffix or '.jpg'
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    file.save(tmp.name)
                    temp_path = tmp.name
                detections = get_pipeline().predict_image(temp_path)
                if detections:
                    best = max(detections, key=lambda d: d.detection_confidence)
                    plate = best.text or 'unreadable'
                    message = f'Plate: {plate} | Detection confidence: {best.detection_confidence:.2f} | OCR confidence: {best.ocr_confidence:.2f}'
                else:
                    message = 'No license plate detected.'
            except Exception as exc:
                message = f'Inference could not run: {exc}'
            finally:
                if temp_path:
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except OSError:
                        pass
    return render_template_string(HTML, message=message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '10000')))
