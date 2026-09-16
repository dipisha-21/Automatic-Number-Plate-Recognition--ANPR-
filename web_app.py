from flask import Flask, request, render_template_string
from pathlib import Path
import tempfile

app = Flask(__name__)

HTML = '''
<!doctype html><html><head><title>ANPR Demo</title>
<style>body{font-family:Arial;max-width:900px;margin:40px auto;padding:0 20px;background:#f6f8fb;color:#20242b}.card{background:#fff;padding:24px;border-radius:14px;box-shadow:0 4px 16px #0001}.result{margin-top:20px;padding:16px;background:#f0f4ff;border-radius:10px}input,button{padding:10px;margin-top:10px}button{cursor:pointer}</style></head><body>
<h1>Automatic Number Plate Recognition</h1><p>YOLOv8 + EasyOCR portfolio demo.</p><div class="card">
<form method="post" enctype="multipart/form-data"><input type="file" name="image" accept="image/*" required><br><button type="submit">Run ANPR</button></form>
{% if message %}<div class="result">{{ message }}</div>{% endif %}
</div><p style="margin-top:18px">Model inference requires a trained weights file configured on the server. This deployment verifies the web interface and app packaging; the repository contains the full pipeline and evaluation code.</p></body></html>
'''


@app.route('/', methods=['GET', 'POST'])
def home():
    message = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            message = 'Please choose an image.'
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                file.save(tmp.name)
                path = tmp.name
            model_path = Path('models/best.pt')
            if not model_path.exists():
                message = 'Demo UI is live. Add trained weights at models/best.pt to enable inference.'
            else:
                try:
                    from src.anpr.pipeline import ANPRPipeline
                    pipeline = ANPRPipeline(str(model_path), gpu=False)
                    detections = pipeline.predict_image(path)
                    if detections:
                        best = max(detections, key=lambda d: d.detection_confidence)
                        message = f'Plate: {best.text or "unreadable"} | Detection confidence: {best.detection_confidence:.2f} | OCR confidence: {best.ocr_confidence:.2f}'
                    else:
                        message = 'No license plate detected.'
                except Exception as exc:
                    message = f'Inference could not run: {exc}'
    return render_template_string(HTML, message=message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
