from flask import Flask, request, jsonify
import os
import tempfile
from transcription import process_audio_file

app = Flask(__name__)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
        audio_file.save(temp_file.name)
        temp_file_path = temp_file.name

    try:
        # Process the audio file
        result = process_audio_file(temp_file_path)

        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to process audio file'}), 500

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
