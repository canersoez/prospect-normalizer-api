
from flask import Flask, request, send_file
import tempfile
import subprocess
import os

app = Flask(__name__)
NORMALIZER_SCRIPT = "prospect-normalizer-v50.py"

@app.route('/normalize', methods=['POST'])
def normalize():
    uploaded_file = request.files['file']
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.csv")
        output_path = os.path.join(tmpdir, "output.csv")
        uploaded_file.save(input_path)
        subprocess.run(["python3", NORMALIZER_SCRIPT, "--input", input_path, "--output", output_path], check=True)
        return send_file(output_path, as_attachment=True, download_name="normalized.csv", mimetype="text/csv")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
