#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# SpeechBrain is already a dependency via pyannote.audio in most installs.
# If it isn't, you'll see a clear import error.
from speechbrain.inference import EncoderClassifier


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def ffmpeg_extract(input_audio: Path, start: float, end: float, out_wav: Path) -> None:
    # -ss/-to before -i is faster but less accurate; we want accurate cuts -> place after -i
    dur = max(0.0, end - start)
    if dur <= 0.01:
        raise ValueError("Segment too short")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(input_audio),
        "-ss", f"{start:.3f}",
        "-t", f"{dur:.3f}",
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(out_wav),
    ]
    run(cmd)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def load_index(index_path: Path) -> dict:
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return {}


def save_index(index_path: Path, index: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def next_vp_id(index: dict) -> str:
    # VP001, VP002, ...
    nums = []
    for k in index.keys():
        m = re.fullmatch(r"VP(\d{3,})", k)
        if m:
            nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    return f"VP{n:03d}"


def speaker_segments(data: dict) -> dict[str, list[dict]]:
    segs = data.get("segments", [])
    by = {}
    for s in segs:
        spk = s.get("speaker")
        if not spk:
            continue
        by.setdefault(spk, []).append(s)
    return by


def pick_segments(segs: list[dict], max_segments: int, min_dur: float) -> list[dict]:
    # take the longest segments first, ignore tiny ones
    scored = []
    for s in segs:
        st = float(s.get("start", 0.0))
        en = float(s.get("end", 0.0))
        dur = en - st
        if dur >= min_dur:
            scored.append((dur, st, en, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [x[3] for x in scored[:max_segments]]


def embed_wavs(classifier: EncoderClassifier, wavs: list[Path]) -> np.ndarray:
    embs = []
    for w in wavs:
        # SpeechBrain API differs by version.
        if hasattr(classifier, "encode_file"):
            e = classifier.encode_file(str(w))
        else:
            # Older versions use encode_batch + waveform loading helper
            from speechbrain.dataio.dataio import read_audio
            wav = read_audio(str(w)).unsqueeze(0)  # [1, T]
            e = classifier.encode_batch(wav)

        e = e.detach().cpu().numpy().reshape(-1).astype(np.float32)
        embs.append(e)

    if not embs:
        raise RuntimeError("No embeddings computed (no wavs?)")

    m = np.mean(np.stack(embs, axis=0), axis=0)
    m = m / (np.linalg.norm(m) + 1e-9)
    return m.astype(np.float32)


def load_existing_embeddings(emb_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    if not emb_dir.exists():
        return out
    for p in emb_dir.glob("VP*.npy"):
        vp = p.stem
        try:
            out[vp] = np.load(p).astype(np.float32)
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="WhisperX output json, e.g. /pipeline/diarisation/RDA_AGM.json")
    ap.add_argument("--audio", required=True, help="Original audio, e.g. /pipeline/transcripts/RDA_AGM.mp3 or /pipeline/processing/x.mp3")
    ap.add_argument("--out_dir", default="/pipeline/voice_library", help="Voice library root")
    ap.add_argument("--meeting_id", default="", help="Optional name for meeting map file (defaults to json stem)")
    ap.add_argument("--threshold", type=float, default=0.78, help="Cosine sim threshold to reuse an existing VP")
    ap.add_argument("--max_segments", type=int, default=12, help="Max segments to sample per speaker")
    ap.add_argument("--min_segment_seconds", type=float, default=1.2, help="Ignore segments shorter than this")
    args = ap.parse_args()

    json_path = Path(args.json)
    audio_path = Path(args.audio)
    if not json_path.exists():
        print(f"[voiceprints] ERROR json not found: {json_path}", file=sys.stderr)
        sys.exit(2)
    if not audio_path.exists():
        print(f"[voiceprints] ERROR audio not found: {audio_path}", file=sys.stderr)
        sys.exit(2)

    out_root = Path(args.out_dir)
    emb_dir = out_root / "embeddings"
    meetings_dir = out_root / "meetings"
    tmp_dir = Path("/pipeline/processing/tmp_voiceprints")

    emb_dir.mkdir(parents=True, exist_ok=True)
    meetings_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_root / "index.json"
    index = load_index(index_path)

    meeting_id = args.meeting_id.strip() or json_path.stem
    map_path = meetings_dir / f"{meeting_id}.map.json"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_spk = speaker_segments(data)

    if not by_spk:
        print("[voiceprints] No speakers found in JSON. Nothing to do.")
        sys.exit(0)

    print(f"[voiceprints] speakers: {', '.join(sorted(by_spk.keys()))}")

    # SpeechBrain model (public). Uses HF cache. Token not required for this model,
    # but if HF_TOKEN is set, HF libs will use it automatically.
    print("[voiceprints] loading embedding model (speechbrain/spkrec-ecapa-voxceleb)...")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    existing = load_existing_embeddings(emb_dir)

    meeting_map = {
        "meeting_id": meeting_id,
        "created": datetime.utcnow().isoformat() + "Z",
        "threshold": args.threshold,
        "speakers": {}
    }

    for spk, segs in sorted(by_spk.items()):
        chosen = pick_segments(segs, max_segments=args.max_segments, min_dur=args.min_segment_seconds)

        if not chosen:
            print(f"[voiceprints] {spk}: no usable segments (all too short). skipping")
            continue

        wavs = []
        for i, s in enumerate(chosen, start=1):
            st = float(s["start"])
            en = float(s["end"])
            out_wav = tmp_dir / f"{meeting_id}_{spk}_{i:02d}.wav"
            ffmpeg_extract(audio_path, st, en, out_wav)
            wavs.append(out_wav)

        vp_emb = embed_wavs(classifier, wavs)

        best_vp = ""
        best_sim = -1.0
        for vp_id, emb in existing.items():
            sim = cosine(vp_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_vp = vp_id

        if best_vp and best_sim >= args.threshold:
            assigned = best_vp
            action = f"matched {best_vp} (sim={best_sim:.3f})"
        else:
            assigned = next_vp_id(index)
            np.save(emb_dir / f"{assigned}.npy", vp_emb)
            existing[assigned] = vp_emb
            index[assigned] = {
                "name": index.get(assigned, {}).get("name", ""),  # blank until you label it
                "created": datetime.utcnow().isoformat() + "Z",
            }
            action = f"NEW {assigned} (best={best_vp or 'none'} sim={best_sim:.3f})"

        meeting_map["speakers"][spk] = {
            "voiceprint_id": assigned,
            "note": action,
        }
        print(f"[voiceprints] {spk}: {action}")

    save_index(index_path, index)
    map_path.write_text(json.dumps(meeting_map, indent=2), encoding="utf-8")
    print(f"[voiceprints] wrote: {map_path}")
    print(f"[voiceprints] updated: {index_path}")


if __name__ == "__main__":
    main()
