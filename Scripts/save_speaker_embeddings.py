#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


def ffmpeg_to_wav16k_mono(src: Path) -> Path:
    """
    Convert any audio to 16kHz mono WAV using ffmpeg.
    Returns path to temp wav file.
    """
    tmp = tempfile.NamedTemporaryFile(prefix="emb_", suffix=".wav", delete=False)
    tmp.close()
    out = Path(tmp.name)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio (mp3/m4a/wav).")
    parser.add_argument("--whisperx_json", required=True, help="Path to WhisperX diarised JSON (has speaker labels).")
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

    # HF token (same pattern as your run_once.sh)
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("[emb] WARNING: HF_TOKEN env var not set. If pyannote models are gated, downloads will fail.")

    # --- Load diarised segments ---
    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = [s for s in data.get("segments", []) if s.get("speaker") is not None]

    if not segments:
        print("[emb] No diarised segments found in JSON (no 'speaker' fields).")
        return

    segments.sort(key=lambda s: float(s.get("start", 0.0)))

    # --- Ensure 16k mono WAV for reliable slicing ---
    tmp_wav = None
    try:
        if audio_path.suffix.lower() != ".wav":
            tmp_wav = ffmpeg_to_wav16k_mono(audio_path)
            wav_path = tmp_wav
        else:
            # Even if it's WAV, still enforce 16k mono if needed.
            audio_info = sf.info(str(audio_path))
            if audio_info.samplerate != 16000 or audio_info.channels != 1:
                tmp_wav = ffmpeg_to_wav16k_mono(audio_path)
                wav_path = tmp_wav
            else:
                wav_path = audio_path

        audio_np, sr = sf.read(str(wav_path), always_2d=True)  # (samples, channels)
        audio_mono = audio_np.mean(axis=1).astype("float32")  # (samples,)
        waveform = torch.from_numpy(audio_mono).unsqueeze(0)  # (1, samples)

        if sr != 16000:
            print(f"[emb] WARNING: sample rate is {sr}Hz (expected 16000Hz).")

        # --- Load embedding model ---
        device = torch.device("cpu")

        # Try to pass token if supported; otherwise rely on env var
        try:
            embedder = PretrainedSpeakerEmbedding("pyannote/embedding", device=device, use_auth_token=hf_token or None)
        except TypeError:
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

            if chunk.shape[1] < int(args.min_seg_s * sr):
                continue

            with torch.inference_mode():
                emb = embedder(chunk)  # pyannote 3.x expects waveform tensor only

            speaker_embeddings.setdefault(speaker, []).append(np.asarray(emb, dtype=np.float32))

        print(f"[emb] Speakers (diarised labels): {len(set(s.get('speaker') for s in segments))}")
        print(f"[emb] Speakers with embeddings: {len(speaker_embeddings)}")

        # --- Save averaged embeddings per speaker ---
        saved = 0
        for speaker, embs in speaker_embeddings.items():
            if len(embs) < args.min_segs_per_speaker:
                print(f"[emb] Skipping {speaker}: only {len(embs)} segments")
                continue

            avg = np.mean(np.stack(embs, axis=0), axis=0)
            out_path = out_dir / f"{speaker}.npy"
            np.save(out_path, avg)
            saved += 1
            print(f"[emb] Saved {out_path.name} ({len(embs)} segments)")

        print(f"[emb] Done. Saved {saved} speaker embeddings to: {out_dir}")

    finally:
        if tmp_wav:
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
