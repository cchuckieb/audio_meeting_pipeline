#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="WhisperX diarised JSON")
    ap.add_argument("--map", required=True, help="Meeting's voice_library/meetings/<id>.map.json")
    ap.add_argument("--index", required=True, help="voice_library/index.json")
    ap.add_argument("--out", required=True, help="Output path for the named transcript .txt")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    meeting_map = json.loads(Path(args.map).read_text(encoding="utf-8")) if Path(args.map).exists() else {"speakers": {}}
    index = json.loads(Path(args.index).read_text(encoding="utf-8")) if Path(args.index).exists() else {}

    speakers = meeting_map.get("speakers", {})

    def resolve_name(spk: str) -> str:
        entry = speakers.get(spk)
        if not entry or not entry.get("voiceprint_id"):
            return spk  # skipped (too little audio) or never voiceprinted
        vp = entry["voiceprint_id"]
        name = index.get(vp, {}).get("name", "").strip()
        return name if name else vp  # fall back to the VP### until someone names it

    segments = [s for s in data.get("segments", []) if s.get("speaker")]
    segments.sort(key=lambda s: float(s.get("start", 0.0)))

    lines = []
    for s in segments:
        text = s.get("text", "").strip()
        if not text:
            continue
        name = resolve_name(s["speaker"])
        ts = format_timestamp(float(s.get("start", 0.0)))
        lines.append(f"[{ts}] {name}: {text}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[label_transcript] wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
