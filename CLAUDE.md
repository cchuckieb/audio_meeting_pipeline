# Audio meeting pipeline

Board meeting transcription pipeline running on a Synology NAS (DXP2800, Intel N100, 8GB RAM, no GPU).

## What it does

1. A separate folder-monitor job detects a new recording in `Incoming/` and kicks off WhisperX
2. WhisperX transcribes and diarises the audio → outputs JSON/SRT/VTT/TSV to `Diarisation/`
3. `Scripts/save_speaker_embeddings.py` extracts per-speaker voice embeddings using pyannote → saves `SPEAKER_XX.npy` to `Voice_Library/`
4. User currently pastes the transcript into ChatGPT manually to produce minutes

## Folder structure

```
Audio/
├── Incoming/          # Drop M4A/WAV recordings here (gitignored)
├── Processed/         # Intermediate WAV files (gitignored)
├── Diarisation/       # WhisperX output: JSON, SRT, VTT, TSV (gitignored)
├── Transcripts/       # Final transcripts (gitignored)
├── Voice_Library/     # Speaker embeddings as .npy files (gitignored)
└── Scripts/
    └── save_speaker_embeddings.py   # Main script
```

## Environment

- Conda env: `whisper` at `C:\Users\camer\miniforge3\envs\whisper\`
- Activate: `conda activate whisper`
- Key deps: torch, soundfile, pyannote.audio, numpy

## Running the script

```bash
python Scripts/save_speaker_embeddings.py \
  --audio "Incoming/meeting.wav" \
  --whisperx_json "Diarisation/meeting.json" \
  --out_dir "Voice_Library"
```

## Improvement plan

Work through phases in order. Update this file when a phase is complete.

### Phase 1 — Foundation (not started)
- [ ] 1. `requirements.txt` — pin all deps
- [ ] 2. Logging — timestamped log file per run
- [ ] 3. Error handling — skip bad segments, handle malformed JSON gracefully
- [ ] 4. Auto-resample non-16kHz audio (dep: librosa or scipy)

### Phase 2 — Named speaker library (not started)
- [ ] 5. Named voice library — enrol speakers by name; user will rename SPEAKER_XX.npy files from a known meeting
- [ ] 6. Speaker matching — cosine similarity against named library, output SPEAKER_XX → name + confidence

### Phase 3 — Minutes generation (not started)
- [ ] 7. Named transcript output — rewrite WhisperX transcript replacing SPEAKER_XX with real names
- [ ] 8. Automated minutes via Claude API — send named transcript to Claude, save as .docx/.md

## Key decisions
- CPU only — no GPU on NAS, overnight runs are fine, Whisper large model is acceptable
- Board attendees are mostly consistent; matcher must output "no match" for low-confidence/unknown speakers
- Named library bootstrapped from existing recordings where attendees are known (rename SPEAKER_XX.npy manually)
- WhisperX step runs separately; this repo handles post-diarisation processing only
