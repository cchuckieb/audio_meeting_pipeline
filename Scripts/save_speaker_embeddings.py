import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--whisperx_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    json_path = Path(args.whisperx_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load WhisperX JSON
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = [s for s in data.get("segments", []) if s.get("speaker")]

    if not segments:
        print("No diarised segments found.")
        return

    # Load audio
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    device = torch.device("cpu")
    embedder = PretrainedSpeakerEmbedding("pyannote/embedding", device=device)

    speaker_embeddings = {}

    for seg in segments:
        speaker = seg["speaker"]
        start = float(seg["start"])
        end = float(seg["end"])

        if end - start < 1.0:
            continue

        start_sample = int(start * sr)
        end_sample = int(end * sr)

        chunk = waveform[:, start_sample:end_sample]

        if chunk.shape[1] < sr:
            continue

        with torch.inference_mode():
            emb = embedder(chunk, sample_rate=sr)

        speaker_embeddings.setdefault(speaker, []).append(
            np.asarray(emb, dtype=np.float32)
        )

    for speaker, embs in speaker_embeddings.items():
        if len(embs) < 3:
            continue

        avg = np.mean(np.stack(embs), axis=0)
        np.save(out_dir / f"{speaker}.npy", avg)

    print("Done. Embeddings saved.")


if __name__ == "__main__":
    main()