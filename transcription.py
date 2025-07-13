import os
import json
import numpy as np
import librosa
import whisper
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()


def load_audio_file(file_path, target_sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None


def process_audio_file(audio_file_path):
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' not found.")
        return None

    audio, sr = load_audio_file(audio_file_path, target_sr=16000)
    if audio is None:
        return None

    print(f"Processing audio file: {audio_file_path}")
    print(f"Audio duration: {len(audio) / sr:.2f} seconds")

    pipeline = Pipeline.from_pretrained(
        "collinbarnwell/pyannote-speaker-diarization-31",
        use_auth_token=os.getenv("HUGGINGFACE_API_KEY"))

    pipeline.to(torch.device("cuda"))

    diarization = pipeline(audio_file_path)

    model = whisper.load_model("base")

    diarization_results = []

    print("\nSpeaker diarization with transcription:")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)

        segment = audio[start_sample:end_sample]

        if len(segment) > 0:
            segment = segment.astype(np.float32)

            result = model.transcribe(segment)
            transcription = result["text"].strip()

            print(
                f"Speaker {speaker} [{turn.start:.1f}s - {turn.end:.1f}s]: {transcription}")

            segment_data = {
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "text": transcription
            }
            diarization_results.append(segment_data)

    final_json = {
        "audio_file": os.path.basename(audio_file_path),
        "file_format": os.path.splitext(audio_file_path)[1].lower(),
        "segments": diarization_results,
        "total_segments": len(diarization_results)
    }

    return final_json


if __name__ == "__main__":
    audio_file = "song.mp3"

    result = process_audio_file(audio_file)

    if result:
        print("\nJSON Output:")
        print(json.dumps(result, indent=2))

        output_filename = os.path.splitext(
            audio_file)[0] + "_transcription.json"
        with open(output_filename, "w") as json_file:
            json.dump(result, json_file, indent=2)

        print(f"\nTranscription saved to: {output_filename}")
    else:
        print("Failed to process audio file.")
