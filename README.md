# Audio Transcription with Speaker Diarization

A Flask API that provides audio transcription with speaker diarization using OpenAI's Whisper and pyannote.audio. The service processes audio files to identify different speakers and transcribe their speech segments.

## Features

- **Speaker Diarization**: Identifies different speakers in audio files
- **Speech Transcription**: Converts speech to text using OpenAI Whisper
- **REST API**: Simple Flask API endpoint for easy integration
- **Multiple Audio Formats**: Supports various audio formats (MP3, WAV,)
- **GPU Support**: Utilizes CUDA for faster processing when available

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- HuggingFace API key

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd diarization-whisper
    ```

2. Install dependencies:

    ```bash
    pip install flask torch torchaudio librosa openai-whisper pyannote.audio python-dotenv numpy
    ```

3. Set up environment variables:

    - Create a `.env` file in the project root:

      ```
      HUGGINGFACE_API_KEY=your_huggingface_api_key_here
      ```

    - Get HuggingFace API key:
      - Sign up at [HuggingFace](https://huggingface.co/)
      - Go to Settings → Access Tokens
      - Create a new token with read permissions
      - Accept the terms for the `pyannote/speaker-diarization-3.1` model
      - Add the token to your `.env` file

## Usage

### Running the API

Start the Flask server:

```bash
python app.py
```

The API will be available at [http://localhost:5000](http://localhost:5000)

### API Endpoint

**POST** `/transcribe`

Upload an audio file and receive transcription with speaker diarization.

#### Request

- Method: POST
- Content-Type: multipart/form-data
- Body: Audio file in `audio` field

#### Response

```json
{
  "audio_file": "example.mp3 || example.wav",
  "file_format": ".mp3",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.5,
      "end": 3.2,
      "text": "Hello, how are you today?"
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.5,
      "end": 6.1,
      "text": "I'm doing well, thank you for asking."
    }
  ],
  "total_segments": 2
}
```

#### Error Response

```json
{
  "error": "No audio file provided"
}
```

### Example Usage

**Using curl:**

```bash
curl -X POST -F "audio=@path/to/your/audio.mp3" http://localhost:5000/transcribe
```

**Using Python requests:**

```python
import requests

with open('audio.mp3', 'rb') as f:
    response = requests.post('http://localhost:5000/transcribe',
                             files={'audio': f})
    result = response.json()
    print(result)
```

### Standalone Script

You can also run the transcription directly:

```bash
python transcription.py
```

Edit the `audio_file` variable in the script to specify your audio file path. The output will be saved as `{filename}_transcription.json`.

## Configuration

### Audio Processing

- Target sample rate: 16kHz
- Whisper model: Base (can be changed in [`transcription.py`](transcription.py))
- Diarization model: `collinbarnwell/pyannote-speaker-diarization-31`

### Server Configuration

- Host: 0.0.0.0 (accessible from network)
- Port: 5000
- Debug mode: Disabled (can be enabled for development)

## Error Handling

The API handles various error conditions:

- Missing audio file in request (400)
- Empty filename (400)
- Audio processing failures (500)
- File loading errors

## Performance Notes

- **GPU Acceleration**: The system uses CUDA when available for faster processing
- **Model Loading**: Whisper model is loaded for each request (consider caching for production)
- **Temporary Files**: Audio files are temporarily saved and cleaned up after processing
- **Memory Usage**: Large audio files may require significant memory

## Supported Audio Formats

- MP3
- WAV
- And other formats supported by `librosa`

## Troubleshooting

### Common Issues

#### HuggingFace Authentication Error

- Verify your API key is correct in the `.env` file
- Ensure you've accepted the model terms on HuggingFace

#### CUDA Out of Memory

- Process shorter audio segments
- Use CPU processing by removing the `.to(torch.device("cuda"))` line in [`transcription.py`](transcription.py)

#### Audio Loading Errors

- Check if the audio file format is supported
- Verify the file is not corrupted

## Dependencies

- flask: Web framework
- torch: Deep learning framework
- openai-whisper: OpenAI's speech recognition
- pyannote.audio: Speaker diarization
- librosa: Audio processing
- numpy: Numerical computations
- python-dotenv: Environment variable management