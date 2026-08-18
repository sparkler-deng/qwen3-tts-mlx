"""Gradio web app + REST API for Qwen3-TTS on Apple Silicon."""

import argparse
import io
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import librosa
import mlx_whisper
import numpy as np
import soundfile as sf
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from huggingface_hub import scan_cache_dir, snapshot_download
from pydantic import BaseModel, Field

from tts import (
    VOICES,
    ALL_MODELS,
    PRESET_MODELS,
    DESIGN_MODELS,
    CLONE_MODELS,
    LANGUAGES,
    SAVED_VOICES_DIR,
    _lock,
    get_model_status,
    load_saved_voices,
    generate_preset_audio,
    generate_design_audio,
    generate_clone_audio,
)

INSTRUCT_PRESETS = [
    "speak with excitement and enthusiasm",
    "slow deliberate pace with dramatic pauses",
    "steady speed, clear articulation",
    "whispered, secretive tone",
]

VOICE_DESIGN_PRESETS = [
    "wise elderly mentor, warm and reassuring, measured pace",
    "energetic young narrator, bright and enthusiastic",
    "calm professional news anchor, clear and authoritative",
    "friendly storyteller, expressive with gentle warmth",
]

# OpenAI speech-API voice names → closest local preset voice.
OPENAI_VOICE_ALIASES = {
    "alloy": "Aiden",
    "ash": "Ryan",
    "ballad": "Serena",
    "coral": "Vivian",
    "echo": "Ryan",
    "fable": "Serena",
    "nova": "Vivian",
    "onyx": "Uncle_Fu",
    "sage": "Sohee",
    "shimmer": "Sohee",
    "verse": "Dylan",
}

OPENAI_MODEL_ALIASES = {"tts-1", "tts-1-hd", "gpt-4o-mini-tts"}

# response_format → (soundfile format, subtype, media type). "pcm" is handled
# separately as raw 16-bit little-endian samples.
SPEECH_FORMATS = {
    "mp3": ("MP3", None, "audio/mpeg"),
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/pcm"),
}

# Local Whisper STT models (name → HF repo) for /v1/audio/transcriptions.
# The model auto-downloads on first use and stays resident (mlx_whisper keeps
# a single cached model, swapped like the TTS side).
STT_MODELS = {
    "whisper-turbo": "mlx-community/whisper-turbo",
    "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-small": "mlx-community/whisper-small-mlx",
    "whisper-base": "mlx-community/whisper-base-mlx",
}
DEFAULT_STT_MODEL = "whisper-turbo"

# OpenAI transcription model names accepted as the local default.
STT_MODEL_ALIASES = {"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"}

# Match DeepTutor / OpenAI clients: audio uploads are capped at 25 MB.
MAX_STT_FILE_BYTES = 25 * 1024 * 1024


def find_preset_voice(name: str) -> str | None:
    """Return the full preset label matching a voice name, case-insensitive."""
    wanted = name.strip().lower()
    for v in VOICES:
        if v.lower() == wanted or v.split(" (")[0].lower() == wanted:
            return v
    return None


def openai_error(status: int, message: str, err_type: str = "invalid_request_error", code: str | None = None):
    """Build an OpenAI-style error response."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )


def encode_speech_audio(audio: np.ndarray, response_format: str) -> tuple[bytes, str]:
    """Encode float mono 24 kHz audio for an OpenAI speech response_format."""
    if response_format == "pcm":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        return pcm.tobytes(), "audio/pcm"
    sf_format, subtype, media_type = SPEECH_FORMATS[response_format]
    buf = io.BytesIO()
    if subtype:
        sf.write(buf, audio, 24000, format=sf_format, subtype=subtype)
    else:
        sf.write(buf, audio, 24000, format=sf_format)
    return buf.getvalue(), media_type


def transcribe_audio_file(audio_path: str, model_name: str, language: str | None) -> str:
    """Transcribe an audio file with the local Whisper model.

    Shares the TTS generation lock so GPU memory is only used by one model
    pass at a time.
    """
    repo = STT_MODELS.get(model_name, STT_MODELS[DEFAULT_STT_MODEL])
    kwargs = {}
    if language and language.strip().lower() not in ("", "auto"):
        kwargs["language"] = language.strip()
    with _lock:
        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=repo, **kwargs)
    if isinstance(result, dict):
        return (result.get("text") or "").strip()
    return str(getattr(result, "text", "")).strip()


def download_model(repo_id: str) -> str:
    """Download model weights into the HuggingFace cache."""
    snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.tiktoken"])
    return f"Downloaded {repo_id}"


def delete_model(repo_id: str) -> str:
    """Delete model from cache."""
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            revision_hashes = [rev.commit_hash for rev in repo.revisions]
            strategy = cache_info.delete_revisions(*revision_hashes)
            strategy.execute()
            return f"Deleted {repo_id}"
    return f"Model {repo_id} not found in cache"


def save_cloned_voice(audio_path: str, transcript: str, name: str) -> str:
    """Persist a cloned voice to saved_voices/."""
    if not name.strip():
        raise gr.Error("Please enter a name for the voice")
    if not audio_path:
        raise gr.Error("Please upload reference audio first")
    if not transcript.strip():
        raise gr.Error("Please enter the transcript first")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    voice_dir = SAVED_VOICES_DIR / safe_name

    if voice_dir.exists():
        raise gr.Error(f"Voice '{safe_name}' already exists")

    voice_dir.mkdir(parents=True)

    original_ext = Path(audio_path).suffix or ".wav"
    shutil.copy(audio_path, voice_dir / f"audio{original_ext}")

    (voice_dir / "transcript.txt").write_text(transcript.strip())
    (voice_dir / "metadata.json").write_text(json.dumps({
        "name": name.strip(),
        "created": datetime.now().isoformat(),
    }))

    return safe_name


def save_designed_voice(audio_path: str, transcript: str, instruct: str, name: str) -> str:
    """Save a designed voice for reuse in voice cloning."""
    if not name.strip():
        raise gr.Error("Please enter a name for the voice")
    if not audio_path:
        raise gr.Error("Please generate audio first")
    if not transcript.strip():
        raise gr.Error("No transcript available")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    voice_dir = SAVED_VOICES_DIR / safe_name

    if voice_dir.exists():
        raise gr.Error(f"Voice '{safe_name}' already exists")

    voice_dir.mkdir(parents=True)

    shutil.copy(audio_path, voice_dir / "audio.wav")

    (voice_dir / "transcript.txt").write_text(transcript.strip())
    (voice_dir / "metadata.json").write_text(json.dumps({
        "name": name.strip(),
        "created": datetime.now().isoformat(),
        "voice_description": instruct.strip() if instruct else "",
        "type": "designed",
    }))

    return safe_name


def delete_saved_voice(name: str) -> str:
    """Delete a saved voice directory from saved_voices/."""
    if name not in load_saved_voices():
        raise gr.Error(f"Voice '{name}' not found")
    shutil.rmtree(SAVED_VOICES_DIR / name)
    return name


def rename_generation(history: list, index: int, new_name: str) -> list:
    """Rename a file on disk and update history."""
    if index < 0 or index >= len(history):
        return history

    entry = history[index]
    old_path = Path(entry["path"])

    new_name = Path(new_name).name

    if not new_name.endswith(".wav"):
        new_name = new_name + ".wav"

    if not new_name or new_name == ".wav":
        raise gr.Error("Invalid filename")

    new_path = old_path.parent / new_name

    if new_path.exists() and new_path != old_path:
        raise gr.Error(f"File '{new_name}' already exists")

    if old_path.exists():
        old_path.rename(new_path)

    history[index]["path"] = str(new_path)
    history[index]["filename"] = new_name

    return history


def delete_generation(history: list, index: int, delete_file: bool) -> list:
    """Remove entry from history and optionally delete file from disk."""
    if index < 0 or index >= len(history):
        return history

    entry = history[index]
    if delete_file:
        filepath = Path(entry["path"])
        if filepath.exists():
            filepath.unlink()

    return history[:index] + history[index + 1:]


def generate_preset(
    text: str, voice: str, instruct: str, temp: float, model_name: str, history: list
) -> tuple:
    """Generate audio using preset voice (Gradio wrapper)."""
    try:
        _audio, metadata = generate_preset_audio(text, voice, instruct, temp, model_name)
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e))

    new_history = [metadata] + history
    return metadata["path"], new_history


def generate_design(
    text: str, instruct: str, language: str, temp: float, history: list
) -> tuple:
    """Generate audio using VoiceDesign model (Gradio wrapper)."""
    try:
        _audio, metadata = generate_design_audio(text, instruct, language, temp)
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e))

    new_history = [metadata] + history
    return metadata["path"], new_history


def generate_clone(
    text: str,
    saved_voice: str,
    temp: float,
    model_name: str,
    history: list,
) -> tuple:
    """Generate audio using voice cloning (Gradio wrapper)."""
    try:
        _audio, metadata = generate_clone_audio(text, saved_voice, temp, model_name)
    except (ValueError, RuntimeError) as e:
        raise gr.Error(str(e))

    new_history = [metadata] + history
    return metadata["path"], new_history


def get_saved_voice_choices():
    """Get list of saved voice names for dropdown."""
    voices = load_saved_voices()
    return list(voices.keys())


def build_metadata_str(entry: dict) -> str:
    """Build metadata string for display."""
    voice_label = f"Clone: {entry['voice']}" if entry.get("is_clone") else entry["voice"]
    parts = [voice_label, f"T={entry['temperature']}"]
    if entry["instruct"]:
        instruct_short = entry["instruct"][:30] + "..." if len(entry["instruct"]) > 30 else entry["instruct"]
        parts.append(instruct_short)
    return " | ".join(parts)


def refresh_all_slots(history: list):
    """Refresh all 5 output slots based on history state."""
    updates = []
    for i in range(5):
        if i < len(history):
            entry = history[i]
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=entry["filename"]))
            updates.append(gr.update(value=entry["path"], label=build_metadata_str(entry)))
        else:
            updates.append(gr.update(visible=(i == 0)))
            updates.append(gr.update(value=""))
            updates.append(gr.update(value=None, label=""))
    return updates


def refresh_slots_for_shift(history: list):
    """Prepare slots for shift before generation."""
    updates = []
    updates.append(gr.update(visible=True))
    updates.append(gr.update(value=""))
    updates.append(gr.update(value=None, label="Generating..."))

    for i in range(4):
        if i < len(history):
            entry = history[i]
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=entry["filename"]))
            updates.append(gr.update(value=entry["path"], label=build_metadata_str(entry)))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=""))
            updates.append(gr.update(value=None, label=""))
    return updates


CSS = """
.compact-input textarea { font-size: 14px !important; }
.compact-input input { font-size: 14px !important; }
.preset-btn { min-width: 0 !important; padding: 4px 8px !important; font-size: 12px !important; }
.history-section { border-left: 2px solid #444; padding-left: 16px; }
.generate-btn { margin-top: 8px !important; }
.filename-row .progress-bar, .filename-row .progress-text, .filename-row .eta-bar,
.filename-row .wrap, .filename-row .generating { display: none !important; }
.icon-btn { min-width: 36px !important; max-width: 36px !important; min-height: 42px !important; padding: 4px !important; font-size: 20px !important; }
.icon-btn-divider { border-right: 1px solid #666 !important; }
.filename-row { align-items: stretch !important; }
.filename-row > div { display: flex !important; align-items: stretch !important; }
"""

with gr.Blocks(title="Qwen3-TTS") as app:
    history_state = gr.State([])
    # Last Voice Design generation (path + inputs at generation time), so
    # "Save as Voice" never grabs audio produced by another tab.
    design_last_state = gr.State(None)

    gr.Markdown("# Qwen3-TTS")

    with gr.Row():
        # LEFT COLUMN - Controls
        with gr.Column(scale=1):
            with gr.Tabs():
                # PRESET VOICES TAB
                with gr.Tab("Preset Voices"):
                    preset_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to synthesize...",
                        lines=4,
                        elem_classes=["compact-input"],
                    )

                    preset_voice = gr.Dropdown(
                        choices=VOICES,
                        value=VOICES[0],
                        label="Voice",
                    )

                    preset_model = gr.Dropdown(
                        choices=list(PRESET_MODELS.keys()),
                        value=list(PRESET_MODELS.keys())[0],
                        label="Model",
                    )

                    gr.Markdown("**Style presets**", elem_id="style-label")
                    with gr.Row():
                        preset_btns = []
                        for preset in INSTRUCT_PRESETS:
                            btn = gr.Button(preset, size="sm", elem_classes=["preset-btn"])
                            preset_btns.append(btn)

                    preset_instruct = gr.Textbox(
                        label="Style instruction",
                        placeholder="e.g., excited and happy...",
                        lines=1,
                        elem_classes=["compact-input"],
                    )

                    for btn, preset in zip(preset_btns, INSTRUCT_PRESETS):
                        btn.click(fn=lambda p=preset: p, outputs=preset_instruct)

                    preset_temp = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        label="Temperature",
                    )

                    preset_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])

                # VOICE DESIGN TAB
                with gr.Tab("Voice Design"):
                    design_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to synthesize...",
                        lines=4,
                        elem_classes=["compact-input"],
                    )

                    gr.Markdown("**Voice description presets**")
                    with gr.Row():
                        design_preset_btns = []
                        for preset in VOICE_DESIGN_PRESETS:
                            btn = gr.Button(preset, size="sm", elem_classes=["preset-btn"])
                            design_preset_btns.append(btn)

                    design_instruct = gr.Textbox(
                        label="Voice Description",
                        placeholder="e.g., deep male voice with British accent, calm and authoritative...",
                        lines=3,
                        elem_classes=["compact-input"],
                    )

                    for btn, preset in zip(design_preset_btns, VOICE_DESIGN_PRESETS):
                        btn.click(fn=lambda p=preset: p, outputs=design_instruct)

                    design_language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="Auto",
                        label="Language",
                    )

                    design_temp = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.9,
                        step=0.05,
                        label="Temperature",
                    )

                    design_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])

                    with gr.Accordion("Save as Voice", open=False):
                        design_save_name = gr.Textbox(
                            label="Voice name",
                            placeholder="Name for this voice...",
                            elem_classes=["compact-input"],
                        )
                        design_save_btn = gr.Button("Save Voice", size="sm")

                # CLONE VOICE TAB
                with gr.Tab("Clone Voice"):
                    gr.Markdown("Save a reference audio clip to use for voice cloning.")

                    create_ref_audio = gr.Audio(
                        label="Reference audio",
                        type="filepath",
                        sources=["upload"],
                    )
                    create_ref_text = gr.Textbox(
                        label="Transcript",
                        placeholder="Words spoken in reference audio...",
                        lines=3,
                        elem_classes=["compact-input"],
                    )
                    create_voice_name = gr.Textbox(
                        label="Voice name",
                        placeholder="Name for this voice...",
                        elem_classes=["compact-input"],
                    )
                    create_voice_btn = gr.Button("Save Voice", variant="primary", elem_classes=["generate-btn"])

                # VOICE CLONING TAB
                with gr.Tab("Use Saved Voice"):
                    clone_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to synthesize...",
                        lines=4,
                        elem_classes=["compact-input"],
                    )

                    with gr.Row():
                        saved_voice_dropdown = gr.Dropdown(
                            choices=get_saved_voice_choices(),
                            label="Voice",
                            interactive=True,
                            scale=4,
                        )
                        refresh_voice_btn = gr.Button(
                            "↻", scale=0, size="sm",
                            elem_classes=["icon-btn"],
                        )
                        delete_voice_btn = gr.Button(
                            "🗑️", variant="stop", scale=0, size="sm",
                            elem_classes=["icon-btn"],
                        )

                    clone_model = gr.Dropdown(
                        choices=list(CLONE_MODELS.keys()),
                        value=list(CLONE_MODELS.keys())[0],
                        label="Model",
                    )

                    clone_temp = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        label="Temperature",
                    )

                    clone_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])

                # MODELS TAB
                with gr.Tab("Models"):
                    gr.Markdown("### Model Management")
                    gr.Markdown("Download models before use. Models are stored in `~/.cache/huggingface/hub/`")

                    model_status_display = gr.Dataframe(
                        headers=["Model", "Status", "Size"],
                        label="Available Models",
                        interactive=False,
                    )

                    refresh_btn = gr.Button("Refresh Status", variant="primary")

                    with gr.Row():
                        model_selector = gr.Dropdown(
                            choices=list(ALL_MODELS.keys()),
                            label="Select Model",
                        )
                        download_btn = gr.Button("Download", variant="primary")
                        delete_btn_models = gr.Button("Delete", variant="stop")

                    model_output = gr.Textbox(label="Status", interactive=False, visible=False)

        # RIGHT COLUMN - Output
        with gr.Column(scale=1, elem_classes=["history-section"]):
            gr.Markdown("### Output")

            output_slots = []

            for i in range(5):
                is_current = i == 0
                with gr.Group(visible=is_current) as container:
                    with gr.Row(elem_classes=["filename-row"]):
                        filename_box = gr.Textbox(
                            label="",
                            placeholder="Generated Audio" if is_current else f"Previous {i}",
                            scale=4,
                            container=False,
                        )
                        rewind_btn = gr.Button("⏮", size="sm", scale=0, elem_classes=["icon-btn", "icon-btn-divider"])
                        delete_btn = gr.Button("🗑️", size="sm", scale=0, elem_classes=["icon-btn"])
                    audio_player = gr.Audio(
                        type="filepath",
                        label="",
                    )
                output_slots.append({
                    "filename": filename_box,
                    "rewind": rewind_btn,
                    "delete": delete_btn,
                    "audio": audio_player,
                    "container": container,
                })


    # Build flat lists for outputs
    all_slot_outputs = []
    for slot in output_slots:
        all_slot_outputs.extend([slot["container"], slot["filename"], slot["audio"]])

    # Event handlers
    def shift_history(history):
        """Shift history before generating - instant update."""
        return refresh_slots_for_shift(history)

    def do_generate_preset(text, voice, instruct, temp, model, history):
        path, new_history = generate_preset(text, voice, instruct, temp, model, history)
        return [
            new_history,
            gr.update(visible=True),
            gr.update(value=new_history[0]["filename"]),
            gr.update(value=path, label=build_metadata_str(new_history[0])),
        ]

    def do_generate_clone(text, saved_voice, temp, model, history):
        path, new_history = generate_clone(text, saved_voice, temp, model, history)
        return [
            new_history,
            gr.update(visible=True),
            gr.update(value=new_history[0]["filename"]),
            gr.update(value=path, label=build_metadata_str(new_history[0])),
        ]

    def do_generate_design(text, instruct, language, temp, history):
        path, new_history = generate_design(text, instruct, language, temp, history)
        design_last = {"path": path, "text": text, "instruct": instruct}
        return [
            new_history,
            design_last,
            gr.update(visible=True),
            gr.update(value=new_history[0]["filename"]),
            gr.update(value=path, label=build_metadata_str(new_history[0])),
        ]

    def make_rename_handler(slot_index):
        def do_rename(new_name, history):
            if slot_index >= len(history):
                return [history] + refresh_all_slots(history)
            if not new_name.strip():
                return [history] + refresh_all_slots(history)
            new_history = rename_generation(history, slot_index, new_name.strip())
            return [new_history] + refresh_all_slots(new_history)
        return do_rename

    def make_delete_handler(slot_index):
        def do_delete(history):
            if slot_index >= len(history):
                return [history] + refresh_all_slots(history)
            new_history = delete_generation(history, slot_index, delete_file=True)
            return [new_history] + refresh_all_slots(new_history)
        return do_delete

    def do_create_voice(audio, transcript, name):
        safe_name = save_cloned_voice(audio, transcript, name)
        new_choices = get_saved_voice_choices()
        gr.Info(f"Voice '{safe_name}' saved")
        return gr.update(choices=new_choices, value=safe_name)

    def do_save_designed_voice(design_last, name):
        if not design_last:
            raise gr.Error("Please generate audio in the Voice Design tab first")
        audio_path = design_last["path"]
        if not Path(audio_path).exists():
            raise gr.Error("The designed audio file no longer exists. Generate again.")
        safe_name = save_designed_voice(audio_path, design_last["text"], design_last["instruct"], name)
        new_choices = get_saved_voice_choices()
        gr.Info(f"Voice '{safe_name}' saved")
        return gr.update(choices=new_choices, value=safe_name)

    def do_delete_voice(saved_voice):
        if not saved_voice:
            raise gr.Error("Please select a voice to delete")
        delete_saved_voice(saved_voice)
        gr.Info(f"Voice '{saved_voice}' deleted")
        return gr.update(choices=get_saved_voice_choices(), value=None)

    def do_refresh_voices():
        return gr.update(choices=get_saved_voice_choices())

    # Slot 0 outputs (current) for generate
    slot0_outputs = [output_slots[0]["container"], output_slots[0]["filename"], output_slots[0]["audio"]]

    # JavaScript to reset audio seek position
    reset_audio_js = "() => { document.querySelectorAll('audio').forEach(a => { a.currentTime = 0; a.pause(); }); }"

    # In gradio 6, a js callback's return value IS the event payload sent to
    # the server, so it must pass the (single) input value through. Returning
    # nothing would send null inputs; throwing cancels the event.
    def confirm_js(msg: str) -> str:
        return f"(v) => {{ if (!confirm('{msg}')) throw new Error('cancelled'); return [v]; }}"

    # Chain: first shift history (instant), then generate (slow, only updates slot 0), then reset audio
    preset_btn.click(
        fn=shift_history,
        inputs=[history_state],
        outputs=all_slot_outputs,
    ).then(
        fn=do_generate_preset,
        inputs=[preset_text, preset_voice, preset_instruct, preset_temp, preset_model, history_state],
        outputs=[history_state] + slot0_outputs,
    ).then(
        fn=None,
        js=reset_audio_js,
    )

    clone_btn.click(
        fn=shift_history,
        inputs=[history_state],
        outputs=all_slot_outputs,
    ).then(
        fn=do_generate_clone,
        inputs=[clone_text, saved_voice_dropdown, clone_temp, clone_model, history_state],
        outputs=[history_state] + slot0_outputs,
    ).then(
        fn=None,
        js=reset_audio_js,
    )

    design_btn.click(
        fn=shift_history,
        inputs=[history_state],
        outputs=all_slot_outputs,
    ).then(
        fn=do_generate_design,
        inputs=[design_text, design_instruct, design_language, design_temp, history_state],
        outputs=[history_state, design_last_state] + slot0_outputs,
    ).then(
        fn=None,
        js=reset_audio_js,
    )

    # Wire up rename (on blur/submit), rewind, and delete for each slot
    rewind_js = """(e) => {
        let el = e.target;
        while (el && !el.querySelector('audio')) el = el.parentElement;
        if (el) { const a = el.querySelector('audio'); a.currentTime = 0; a.pause(); }
    }"""
    for i, slot in enumerate(output_slots):
        slot["filename"].submit(
            fn=make_rename_handler(i),
            inputs=[slot["filename"], history_state],
            outputs=[history_state] + all_slot_outputs,
        )
        slot["filename"].blur(
            fn=make_rename_handler(i),
            inputs=[slot["filename"], history_state],
            outputs=[history_state] + all_slot_outputs,
        )
        slot["rewind"].click(fn=None, js=rewind_js)
        slot["delete"].click(
            fn=make_delete_handler(i),
            inputs=[history_state],
            outputs=[history_state] + all_slot_outputs,
            js=confirm_js("Delete this audio file from disk?"),
        )

    create_voice_btn.click(
        fn=do_create_voice,
        inputs=[create_ref_audio, create_ref_text, create_voice_name],
        outputs=[saved_voice_dropdown],
    )

    design_save_btn.click(
        fn=do_save_designed_voice,
        inputs=[design_last_state, design_save_name],
        outputs=[saved_voice_dropdown],
    )

    refresh_voice_btn.click(
        fn=do_refresh_voices,
        outputs=[saved_voice_dropdown],
    )

    delete_voice_btn.click(
        fn=do_delete_voice,
        inputs=[saved_voice_dropdown],
        outputs=[saved_voice_dropdown],
        js=confirm_js("Delete this saved voice? This cannot be undone."),
    )

    # Model management handlers
    def refresh_model_status():
        statuses = get_model_status()
        data = []
        for s in statuses:
            status = "✅ Downloaded" if s["downloaded"] else "❌ Not downloaded"
            size = f"{s['size'] / 1e9:.1f} GB" if s["downloaded"] else "—"
            data.append([s["name"], status, size])
        return data

    def do_download_model(model_name):
        if not model_name:
            return gr.update(value="Please select a model", visible=True), refresh_model_status()
        repo_id = ALL_MODELS[model_name]
        result = download_model(repo_id)
        return gr.update(value=result, visible=True), refresh_model_status()

    def do_delete_model(model_name):
        if not model_name:
            return gr.update(value="Please select a model", visible=True), refresh_model_status()
        repo_id = ALL_MODELS[model_name]
        result = delete_model(repo_id)
        return gr.update(value=result, visible=True), refresh_model_status()

    refresh_btn.click(fn=refresh_model_status, outputs=[model_status_display])
    app.load(fn=refresh_model_status, outputs=[model_status_display])
    download_btn.click(
        fn=do_download_model,
        inputs=[model_selector],
        outputs=[model_output, model_status_display],
    )
    delete_btn_models.click(
        fn=do_delete_model,
        inputs=[model_selector],
        outputs=[model_output, model_status_display],
        js=confirm_js("Delete this model from cache? You will need to re-download it to use again."),
    )

# --- REST API (registered on Gradio's FastAPI instance) ---
# launch() replaces app.app with a fresh FastAPI instance, so routes must be
# registered after launch — see register_api() call in __main__.

class PresetRequest(BaseModel):
    text: str
    voice: str = VOICES[0]
    instruct: str = ""
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    model: str = list(PRESET_MODELS.keys())[0]


class DesignRequest(BaseModel):
    text: str
    instruct: str
    language: str = "Auto"
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)


class CloneRequest(BaseModel):
    text: str
    voice: str
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    model: str = list(CLONE_MODELS.keys())[0]


class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech request body."""

    model: str = "tts-1"
    input: str
    voice: str = ""
    instructions: str = ""
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str = "Auto"
    stream: bool = False  # accepted for compatibility; audio arrives in one body


def register_api(fastapi_app):
    """Register the /v1/* REST routes on the served FastAPI instance."""

    @fastapi_app.get("/v1/health")
    def api_health():
        return {"status": "ok"}

    @fastapi_app.get("/v1/voices")
    def api_list_voices():
        preset = [v.split(" (")[0] for v in VOICES]
        saved = list(load_saved_voices().keys())
        return {"preset": preset, "saved": saved}

    @fastapi_app.delete("/v1/voices/{voice_name}")
    def api_delete_voice(voice_name: str):
        if voice_name not in load_saved_voices():
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
        shutil.rmtree(SAVED_VOICES_DIR / voice_name)
        return {"deleted": voice_name}

    @fastapi_app.get("/v1/models")
    def api_list_models():
        statuses = get_model_status()
        return {
            "models": statuses,
            # OpenAI-compatible listing (same response, both shapes).
            "object": "list",
            "data": [
                {"id": s["name"], "object": "model", "created": int(time.time()), "owned_by": "mlx-community"}
                for s in statuses
            ],
        }

    @fastapi_app.post("/v1/tts/generate")
    def api_generate_preset(req: PresetRequest):
        try:
            _audio, metadata = generate_preset_audio(
                text=req.text, voice=req.voice, instruct=req.instruct,
                temperature=req.temperature, model_name=req.model,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return FileResponse(metadata["path"], media_type="audio/wav", filename=metadata["filename"])

    @fastapi_app.post("/v1/tts/design")
    def api_generate_design(req: DesignRequest):
        try:
            _audio, metadata = generate_design_audio(
                text=req.text, instruct=req.instruct,
                language=req.language, temperature=req.temperature,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return FileResponse(metadata["path"], media_type="audio/wav", filename=metadata["filename"])

    @fastapi_app.post("/v1/tts/clone")
    def api_generate_clone(req: CloneRequest):
        try:
            _audio, metadata = generate_clone_audio(
                text=req.text, saved_voice=req.voice,
                temperature=req.temperature, model_name=req.model,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return FileResponse(metadata["path"], media_type="audio/wav", filename=metadata["filename"])

    @fastapi_app.post("/v1/audio/speech")
    def api_create_speech(req: SpeechRequest):
        """OpenAI-compatible text-to-speech.

        Voice resolution order: saved voices, local preset voices,
        OpenAI voice aliases, or "design"/empty for voice design via
        `instructions`.
        """
        fmt = req.response_format.lower()
        if fmt not in SPEECH_FORMATS:
            return openai_error(400, f"response_format must be one of: {', '.join(sorted(SPEECH_FORMATS))}")
        if not req.input.strip():
            return openai_error(400, "input is required")
        if len(req.input) > 4096:
            return openai_error(400, "input must be 4096 characters or less")
        if req.model not in OPENAI_MODEL_ALIASES and req.model not in ALL_MODELS:
            return openai_error(400, f"The model '{req.model}' does not exist", code="model_not_found")

        voice = req.voice.strip()
        saved = load_saved_voices()
        use_design = voice.lower() in ("", "design")
        preset = None if use_design or voice in saved else find_preset_voice(
            OPENAI_VOICE_ALIASES.get(voice.lower(), voice)
        )

        if not (voice in saved or preset or use_design):
            names = [v.split(" (")[0] for v in VOICES]
            return openai_error(
                400,
                f"Voice '{voice}' not found. Use a preset ({', '.join(names)}), "
                f"an OpenAI alias ({', '.join(sorted(OPENAI_VOICE_ALIASES))}), "
                f"a saved voice ({', '.join(saved) or 'none yet'}), or 'design'.",
                code="voice_not_found",
            )

        try:
            if voice in saved:
                if req.model not in OPENAI_MODEL_ALIASES and req.model not in CLONE_MODELS:
                    return openai_error(400, f"Model '{req.model}' cannot clone voices; use {', '.join(CLONE_MODELS)}")
                model_name = req.model if req.model in CLONE_MODELS else "1.7B-Base"
                audio, _metadata = generate_clone_audio(
                    text=req.input, saved_voice=voice, temperature=1.0, model_name=model_name,
                )
            elif use_design:
                if req.model not in OPENAI_MODEL_ALIASES and req.model != "1.7B-VoiceDesign":
                    return openai_error(400, "Model '" + req.model + "' cannot design voices; use 1.7B-VoiceDesign")
                if not req.instructions.strip():
                    return openai_error(400, "instructions (a voice description) are required when voice is 'design'")
                audio, _metadata = generate_design_audio(
                    text=req.input, instruct=req.instructions,
                    language=req.language, temperature=0.9,
                )
            else:
                if req.model not in OPENAI_MODEL_ALIASES and req.model != "1.7B-CustomVoice":
                    return openai_error(400, f"Model '{req.model}' cannot use preset voices; use 1.7B-CustomVoice")
                audio, _metadata = generate_preset_audio(
                    text=req.input, voice=preset, instruct=req.instructions,
                    temperature=1.0, model_name="1.7B-CustomVoice",
                )
        except ValueError as e:
            return openai_error(400, str(e))
        except RuntimeError as e:
            return openai_error(503, str(e), err_type="service_unavailable")

        audio = np.asarray(audio, dtype=np.float32)
        if req.speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=req.speed)
        data, media_type = encode_speech_audio(audio, fmt)
        return Response(content=data, media_type=media_type)

    @fastapi_app.post("/v1/audio/transcriptions")
    def api_transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
    ):
        """OpenAI-compatible speech-to-text using local Whisper models.

        Accepts multipart uploads (any format ffmpeg can decode — webm/opus
        from browsers, mp3, wav, m4a, ...) and returns `{"text": ...}`.
        The model downloads on first use.
        """
        model = (model or "").strip() or "whisper-1"
        if model not in STT_MODEL_ALIASES and model not in STT_MODELS:
            return openai_error(
                400, f"The model '{model}' does not exist", code="model_not_found"
            )
        model_name = model if model in STT_MODELS else DEFAULT_STT_MODEL

        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        try:
            data = file.file.read()
        finally:
            file.file.close()
        if not data:
            return openai_error(400, "file is required")
        if len(data) > MAX_STT_FILE_BYTES:
            return openai_error(413, "file exceeds the 25 MB limit")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            text = transcribe_audio_file(tmp_path, model_name, language)
        except ValueError as e:
            return openai_error(400, str(e))
        except Exception as e:  # model load/download or decode failure
            return openai_error(503, f"Transcription failed: {e}", err_type="service_unavailable")
        finally:
            os.unlink(tmp_path)
        return {"text": text}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    args = parser.parse_args()
    app.launch(server_name=args.host, server_port=args.port, inbrowser=True, prevent_thread_lock=True)
    register_api(app.app)
    app.block_thread()
