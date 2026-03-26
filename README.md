# EchoFoley

## Quick start

### 1. Install dependencies

```bash
pip install openai moviepy tqdm numpy einops torch torchaudio stable-audio-tools pydub
```

Notes:

- `stable-audio-tools` is required for the audio model (`stabilityai/stable-audio-open-small`).
- `moviepy`/`pydub` typically require **ffmpeg** installed system-wide (for video/audio encoding).

### 2. Set environment variables

This code uses OpenRouter for text LLM calls:

- `OPENROUTER_API` (required for `src/text_llm.py`).

It also uses AzureOpenAI-based multimodal calls in `src/video_understanding.py` (and `src/user_query_understanding.py`, not currently used by the main pipeline):

- Those modules currently contain **placeholder `api_key=""`**. To run end-to-end, you must edit the code to provide a valid Azure key (or modify the helper functions to read from environment variables).

### 3. Run the pipeline

From the repo root:

```bash
export PYTHONPATH="./src"
python -c "from pipeline import run; print(run('PATH/TO/video.mp4', \"your instruction\", './output'))"
```

Outputs will be written under `./output/` (see [Outputs](#outputs)).

## How to run modules (entrypoints)

The primary programmatic entrypoint is:

- `src/pipeline.py`: `run(video_path: str, user_instruction: str, output_dir: str) -> str`

`src/pipeline.py` also has a `__main__` demo, but it uses hard-coded video paths and output directories. Prefer calling `run()` as shown above.

## Repository structure

- `src/pipeline.py`: orchestrates registry setup and calls `VideoLLMController.run()`.
- `src/controller.py`: multi-stage controller:
  - reasoning (timestamps + event plan)
  - sound design edits (optional)
  - synthesis (audio generation + mixing + mux back into video)
- `src/reasoning_actions.py`: actions for:
  - `video_overview`
  - `start_timestamp_localization`
  - `end_timestamp_localization`
  - `slow_fast_fusion`
  - `verification_to_event_plan`
- `src/sound_design_actions.py`: action for:
  - `sound_design_edit_plan` (updates the symbolic plan from the user instruction)
- `src/synthesis_actions.py`: actions for:
  - `generation_planner` (turn event plan into backend commands)
  - `generate_audio` (Stable Audio Open Small per event)
  - `tune_audio_volume` (symbolic volume -> dB gain)
  - `mix_audio_tracks` (crossfade + overlay all event tracks)
  - `render_video_with_audio` (muxes mixed audio into the original video)
- `src/video_understanding.py`: multimodal functions used by reasoning to infer sound logs and timestamps.
- `src/text_llm.py`: OpenRouter text-only generation.
- `src/audio_generation.py`: Stable Audio initialization, audio generation, and mixing helpers.
- `tools/mute_video.py`: utility for removing audio from `.mp4` files.

## Event plan schema (the symbolic “(t, d, p)” representation)

The controller represents sounding events as `Event` objects:

- `t`: `(t_start_sec, t_end_sec)` in seconds
- `d`: `EventDescription` with fields:
  - `subject` (e.g., “cat”)
  - `action` (e.g., “meow”)
  - `object` (optional, often empty)
- `p`: `EventProperties` with fields (all strings; default is `"DEFAULT"`):
  - `pitch`
  - `volume`
  - `intensity`
  - `spatial`

Internally, the plan is passed around as JSON via `Event.to_dict()` / `Event.from_dict()`.

Example (illustrative):

```json
{
  "EVENT_PLAN": [
    {
      "id": "evt_0",
      "t": [4.2, 5.8],
      "d": {"subject": "cat", "action": "meow", "object": ""},
      "p": {"pitch": "LOW", "volume": "HIGH", "intensity": "LOW", "spatial": "CENTER"}
    }
  ]
}
```

## What the pipeline does (end-to-end)

### Phase 1: Reasoning (video -> symbolic event plan)

`src/controller.py` runs:

1. `video_overview`: multimodal inference that produces a timestamped log of inferred sound events.
2. `start_timestamp_localization`: refines each event’s **start** time.
3. `end_timestamp_localization`: refines each event’s **end** time (implemented in parallel in `video_understanding.py`).
4. `slow_fast_fusion`: merges/cleans the timelines and produces a consistent merged view.
5. `verification_to_event_plan`: converts merged events into `(t, d, p)` structured event plan.

### Phase 2: Sound design edits (instruction -> updated event plan)

If `user_instruction` is non-empty, the controller runs:

- `sound_design_edit_plan`: asks a text LLM to return a JSON-updated `EVENT_PLAN` after applying edits implied by the instruction.

If parsing/editing fails, it preserves the previous plan (best-effort NO-OP).

The controller validates a basic temporal invariant (strictly increasing `t_start` and `t_end > t_start`).

### Phase 3: Synthesis (event plan -> audio -> final video)

The synthesis phase runs:

1. `generation_planner`: converts each event into a backend “generation command” and mixing instructions (LLM-based; falls back to defaults if parsing fails).
2. `generate_audio`: lazily imports `audio_generation.audio_generation(...)` and generates a `.wav` per event using **Stable Audio Open Small**.
3. `tune_audio_volume`: converts symbolic volume to a numeric gain in dB.
4. `mix_audio_tracks`: crossfades/overlays all generated clips and writes `mixed_audio.wav` and `mixed_audio.mp3`.
5. `render_video_with_audio`: muxes the mixed audio back into the original video and writes `final_video.mp4`.

## Outputs

Given `output_dir`, the pipeline writes:

- `output_dir/time_record.json`: total runtime.
- `output_dir/final_video.mp4`: final muxed video.

It also uses `output_dir/cache/` for intermediate artifacts, including:

- per-event generated audio under `cache_dir/soundtracks/`
- `mixed_audio.wav` / `mixed_audio.mp3`
- intermediate reasoning JSON/text files (overview, timestamps, localization results)

Exact filenames depend on which stages were executed.

## Troubleshooting

1. **Missing `OPENROUTER_API`**
   - Set `OPENROUTER_API` (used by `src/text_llm.py` and planning/edit LLM calls).

2. **Azure multimodal key failures**
   - `src/video_understanding.py` creates `AzureOpenAI(... api_key="")`.
   - You must replace the placeholder key or refactor `t2t_generate`/`v2t_generate` to pull keys from environment variables.

3. **`ModuleNotFoundError: logid`**
   - Some modules import `logid` but it is not present in this repo.
   - If you only need request IDs for logging, replace `logid.generate()` with something like `uuid.uuid4().hex` or install the correct internal package for your environment.

4. **ffmpeg errors**
   - `moviepy` and `pydub` require system `ffmpeg` for encoding/decoding.

5. **GPU / CUDA issues**
   - Stable Audio runs on `cuda:0` if available; otherwise it runs on CPU.
   - For practical runtime, use a CUDA-capable machine.

## Citation

```bibtex
@misc{li2025echofoley,
  title={EchoFoley: Event-Centric Hierarchical cOntrol for Video Grounded Creative Sound Generation},
  author={Li, Bingxuan and Cui, Yiming and He, Yicheng and Wang, Yiwei and Zhang, Shu and Wen, Longyin and Niu, Yulei},
  year={2025},
  eprint={2512.24731},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/pdf/2512.24731}
}
```

# EchoFoley
