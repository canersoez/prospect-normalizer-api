import os
import glob
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, request, send_file, abort

app = Flask(__name__)
NORMALIZER_SCRIPT = "prospect-normalizer-v50.py"

@app.route("/normalize", methods=["POST"])
def normalize():
    """Accept a CSV/XLSX upload -> run normalizer -> return normalized CSV."""
    if "file" not in request.files:
        abort(400, "multipart‑form field 'file' missing")

    uploaded = request.files["file"]
    if uploaded.filename == "":
        abort(400, "empty filename")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / uploaded.filename
        uploaded.save(input_path)

        # Call the normalizer: <script> <input> --out <tmpdir>
        subprocess.run([
            "python3", NORMALIZER_SCRIPT,
            str(input_path),
            "--out", str(tmpdir_path),
            "--quiet"
        ], check=True)

        # Expect output file: normalized_<stem>.csv
        out_pattern = tmpdir_path / f"normalized_{input_path.stem}.csv"
        matches = glob.glob(str(out_pattern))
        if not matches:
            abort(500, "Normalizer did not produce output file")
        output_path = matches[0]

        return send_file(output_path,
                         as_attachment=True,
                         download_name="normalized.csv",
                         mimetype="text/csv")

# Optional health route
@app.get("/healthz")
def health() -> str:
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
