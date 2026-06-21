# Audio meeting pipeline

Board meeting transcription pipeline for RDA Scottish Borders Group, running on a Synology NAS (DXP2800, Intel N100, 8GB RAM, no GPU).

**This repo is code-only.** The live pipeline runs entirely on the NAS at `/volume1/audio_pipeline`, mapped as `Y:` on the admin's Windows PC. All meeting recordings, transcripts, diarisation output, and voice-library data stay off this repo (and off any public location) — see "What's deliberately excluded" below.

## What it does

1. `whisperx_worker` (Docker, `restart: unless-stopped`) runs `Scripts/watch_incoming.sh`, which polls `incoming/` every 20s
2. On a new file: claims it into `processing/`, runs WhisperX (`Scripts/docker/whisperx/run_once.sh`) — transcribes + diarises, model `medium` by default, outputs JSON/SRT/VTT/TSV to `diarisation/`
3. On success: moves audio to `transcripts/`, writes a timestamped log to `logs/`. On failure: moves audio to `failed/` instead, log records the exit status
4. Separately (manual step, not yet wired into the watcher): `Scripts/build_voiceprints.py` extracts a SpeechBrain ECAPA embedding per diarised speaker, matches it against a persistent cross-meeting voice library (cosine similarity, threshold 0.78), and writes a SPEAKER_XX → VP### mapping. Voiceprints are named by hand by editing `voice_library/index.json`
5. User currently pastes the transcript into ChatGPT manually to produce minutes — this is the next real gap (Phase 3 below)

## Folder structure (on the NAS, `/volume1/audio_pipeline`)

```
audio_pipeline/
├── incoming/          # Drop MP3/M4A/WAV recordings here
├── processing/        # In-flight while WhisperX runs
├── transcripts/       # Source recordings, post-success (misleading name — holds audio, not text)
├── diarisation/       # WhisperX output: JSON, SRT, VTT, TSV
├── failed/            # Recordings that failed processing (check logs/)
├── logs/              # One timestamped log per processed file
├── voice_library/      # embeddings/VP###.npy, meetings/<id>.map.json, index.json (VP### -> name)
├── embeddings_raw/     # Output dir for save_speaker_embeddings.py — currently empty, looks superseded by build_voiceprints.py (unconfirmed)
├── cache/, config/.cache/, models/   # HuggingFace/model caches — regenerable, never back these up
└── scripts/
    ├── docker/whisperx/
    │   ├── Dockerfile           # python:3.9-bullseye, pinned whisperx==3.3.1, pyannote.audio==3.1.1, torch==2.2.2, etc.
    │   ├── docker-compose.yml   # whisperx (one-off) + whisperx_worker (always-on watcher)
    │   ├── run_once.sh          # invokes `python -m whisperx`
    │   └── .env                 # HF_TOKEN (gitignored — see .env.example for the shape)
    ├── watch_incoming.sh        # the folder watcher
    ├── save_speaker_embeddings.py   # pyannote/embedding extractor — possibly superseded, output dir is empty
    └── build_voiceprints.py     # SpeechBrain ECAPA cross-meeting speaker matcher — the one actually in use
```

## What's deliberately excluded from this repo (see .gitignore)

The GitHub repo for this project is **public**. Meeting recordings, transcripts, and voice-library data contain real attendees' names and voices, charity financial figures, and discussion of vulnerable program participants (disabled children in the RDA program). None of that may ever be committed here. It's backed up separately to a private location (Google Drive).

## Environment

- Everything runs inside Docker on the NAS — no local conda env needed despite earlier docs suggesting otherwise
- `HF_TOKEN` required (gated pyannote models) — set in `Scripts/docker/whisperx/.env`, never commit the real value

## Running things manually

```bash
# one-off WhisperX run
./Scripts/docker/whisperx/run_once.sh /pipeline/processing/meeting.mp3 medium

# extract + match voiceprints after a meeting's JSON exists
python Scripts/build_voiceprints.py \
  --json /pipeline/diarisation/meeting.json \
  --audio /pipeline/transcripts/meeting.mp3 \
  --out_dir /pipeline/voice_library
```

## Status (corrected 2026-06-21 — previous version of this doc described an earlier, abandoned design)

### Phase 1 — Foundation: done
Achieved differently than originally planned: deps pinned via Dockerfile (not requirements.txt), logging via per-file logs in `watch_incoming.sh`, error handling via `failed/` + exit-status capture, auto-resampling via ffmpeg in `save_speaker_embeddings.py`.

### Phase 2 — Named speaker library: mostly done
- [x] Cross-meeting voice library with cosine-similarity matching (`build_voiceprints.py`)
- [ ] Wire voiceprint extraction into `watch_incoming.sh` so it runs automatically per meeting (currently manual, only run for 2 of 7 meetings so far)
- [ ] Name the remaining voiceprints in `voice_library/index.json` (only VP003 "Susie" is named, out of 17)
- [ ] Investigate low match rate across meetings (only 2 of 9 speakers matched in the one cross-meeting comparison checked) — tune threshold/segment selection, or confirm it's expected attendee turnover

### Phase 3 — Minutes generation: not started
- [ ] Named transcript output — rewrite WhisperX JSON replacing SPEAKER_XX with names from voice_library, output clean .txt
- [ ] Automated minutes via Claude API — send named transcript to Claude, save as .docx/.md, eliminating the manual ChatGPT step

## Key decisions
- CPU only — no GPU on NAS, overnight runs are fine (one meeting took ~1h45m at `medium` model)
- Board attendees are mostly consistent; matcher should output "no match" for low-confidence/unknown speakers (it does — see threshold)
- `save_speaker_embeddings.py` vs `build_voiceprints.py`: two different embedding approaches exist (pyannote/embedding vs SpeechBrain ECAPA). `build_voiceprints.py` is the one with real data behind it; status of `save_speaker_embeddings.py` is unconfirmed with the user
