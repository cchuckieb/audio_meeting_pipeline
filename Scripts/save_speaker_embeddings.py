import argparse
import json
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to WAV audio (recommended 16k mono).")
    parser.add_argument("--whisperx_json", required=True, help="Path to WhisperX diarised JSON.")
    parser.add_argument("--out_dir", required=True, help="Folder to write .npy embeddings.")
    parser.add_argument("--min_seg_s", type=float, default=1.0, help="Ignore segments shorter than this (seconds).")
    parser.add_argument("--min_segs_per_speaker", type=int, default=1, help="Minimum segments required to save a speaker.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    json_path = Path(args.whisperx_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"WhisperX JSON not found: {json_path}")

    # --- Load diarised segments ---
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = [s for s in data.get("segments", []) if s.get("speaker") is not None]

    if not segments:
        print("No diarised segments found in JSON (no 'speaker' fields).")
        return

    segments.sort(key=lambda s: float(s.get("start", 0.0)))

    # --- Load audio (WAV) via soundfile ---
    audio_np, sr = sf.read(str(audio_path), always_2d=True)  # (samples, channels)

    # Convert to mono float32
    audio_mono = audio_np.mean(axis=1).astype("float32")  # (samples,)
    waveform = torch.from_numpy(audio_mono).unsqueeze(0)  # (1, samples)

    if sr != 16000:
        print(f"WARNING: sample rate is {sr}Hz (expected 16000Hz). Consider re-encoding to 16kHz mono WAV.")

    # --- Load embedding model ---
    device = torch.device("cpu")
    embedder = PretrainedSpeakerEmbedding("pyannote/embedding", device=device)

    speaker_embeddings = {}  # speaker -> list[np.ndarray]

    def crop(start_s: float, end_s: float) -> torch.Tensor:
        start = max(0, int(start_s * sr))
        end = max(start + 1, int(end_s * sr))
        return waveform[:, start:end]

    # --- Extract embeddings per diarised speaker label ---
    for seg in segments:
        speaker = seg.get("speaker")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        if not speaker or end <= start:
            continue
        if (end - start) < args.min_seg_s:
            continue

        chunk = crop(start, end)

        # Skip tiny chunks
        if chunk.shape[1] < int(args.min_seg_s * sr):
            continue

        with torch.inference_mode():
            # IMPORTANT: your pyannote version expects ONLY the waveform tensor
            emb = embedder(chunk)

        speaker_embeddings.setdefault(speaker, []).append(np.asarray(emb, dtype=np.float32))

    print(f"Speakers (diarised labels): {len(set(s.get('speaker') for s in segments))}")
    print(f"Speakers with embeddings: {len(speaker_embeddings)}")

    # --- Save averaged embeddings per speaker ---
    saved = 0
    for speaker, embs in speaker_embeddings.items():
        if len(embs) < args.min_segs_per_speaker:
            print(f"Skipping {speaker}: only {len(embs)} segments")
            continue

        avg = np.mean(np.stack(embs, axis=0), axis=0)
        out_path = out_dir / f"{speaker}.npy"
        np.save(out_path, avg)
        saved += 1
        print(f"Saved {out_path.name} ({len(embs)} segments)")

    print(f"Done. Saved {saved} speaker embeddings to: {out_dir}")


if __name__ == "__main__":
    main()