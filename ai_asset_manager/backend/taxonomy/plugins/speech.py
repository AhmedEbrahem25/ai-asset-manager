"""Speech and audio models and datasets."""

from __future__ import annotations

from ai_asset_manager.backend.models.enums import DatasetFormat, ModelType
from ai_asset_manager.backend.taxonomy.plugins._shared import (
    audio_count,
    declared_format,
    family_of,
    is_dataset,
    is_model,
)
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: Speech-recognition families.
ASR_FAMILIES = (
    ("Whisper", ("whisper", "faster-whisper", "distil-whisper")),
    ("Wav2Vec2", ("wav2vec",)),
    ("HuBERT", ("hubert",)),
    ("WavLM", ("wavlm",)),
    ("Parakeet", ("parakeet",)),
    ("Canary", ("nemo-canary", "canary-1b")),
    ("Conformer", ("conformer",)),
    ("Seamless", ("seamless",)),
    ("Moonshine", ("moonshine",)),
)

#: Speech-synthesis families.
TTS_FAMILIES = (
    ("XTTS", ("xtts",)),
    ("Piper", ("piper",)),
    ("Bark", ("suno/bark", "bark-small", "bark_")),
    ("VITS", ("vits",)),
    ("SpeechT5", ("speecht5",)),
    ("Kokoro", ("kokoro",)),
    ("StyleTTS", ("styletts",)),
    ("Tacotron", ("tacotron",)),
    ("F5-TTS", ("f5-tts",)),
    ("Coqui", ("coqui",)),
)

#: Audio corpora.
AUDIO_DATASET_FAMILIES = (
    ("LibriSpeech", ("librispeech",)),
    ("Common Voice", ("common_voice", "common-voice", "commonvoice")),
    ("VoxPopuli", ("voxpopuli",)),
    ("VoxCeleb", ("voxceleb",)),
    ("AudioSet", ("audioset",)),
    ("ESC-50", ("esc-50", "esc50")),
    ("UrbanSound", ("urbansound",)),
    ("GigaSpeech", ("gigaspeech",)),
    ("LJSpeech", ("ljspeech",)),
    ("FLEURS", ("fleurs",)),
)

TASKS = (
    Task(id="speech_recognition", label="Speech Recognition", domain="speech", order=10),
    Task(id="speech_synthesis", label="Speech Synthesis", domain="speech", order=20),
    Task(id="speaker_recognition", label="Speaker Recognition", domain="speech", order=30),
    Task(id="keyword_spotting", label="Keyword Spotting", domain="speech", order=40),
    Task(id="emotion_recognition", label="Emotion Recognition", domain="speech", order=50),
    Task(id="audio_classification", label="Audio Classification", domain="audio", order=60),
    Task(id="audio_generation", label="Audio Generation", domain="audio", order=70),
    Task(id="source_separation", label="Source Separation", domain="audio", order=80),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register speech and audio categories, tasks and classifiers."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="speech", label="Speech", section="models", order=120, domain="speech",
                 aliases=("asr", "speech-recognition"))
    )
    registry.add_category(
        Category(id="text_to_speech", label="Text-to-Speech", section="models", order=130,
                 domain="speech", aliases=("tts", "speech-synthesis"))
    )
    registry.add_category(
        Category(id="audio", label="Audio Model", section="models", order=140, domain="audio",
                 aliases=("audio-models",))
    )
    registry.add_category(
        Category(id="audio_dataset", label="Audio Dataset", section="datasets", order=260,
                 domain="audio", aliases=("audio-datasets", "speech-datasets"))
    )

    # Asking for "speech" means the whole pipeline, not one half of it.
    registry.add_alias("speech", ("speech", "text_to_speech", "audio"))

    registry.add_classifier(_tts_model, name="speech.tts", priority=520)
    registry.add_classifier(_asr_model, name="speech.asr", priority=510)
    registry.add_classifier(_audio_model, name="speech.audio", priority=500)
    registry.add_classifier(_audio_dataset, name="speech.dataset", priority=490)


def _tts_model(profile: AssetProfile) -> Classification | None:
    """Claim speech synthesis models.

    Ahead of recognition because several synthesis stacks embed a recogniser for forced
    alignment, and their names mention both.
    """
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    family = family_of(profile, TTS_FAMILIES)
    marker = profile.matches(("text-to-speech", "tts", "voice-clon", "vocoder"))

    if declared != ModelType.TEXT_TO_SPEECH and family is None and marker is None:
        return None

    return Classification(
        category="text_to_speech", task="speech_synthesis", domain="speech", family=family,
        modalities=("audio", "text"),
        confidence=CONFIDENCE_CERTAIN if declared == ModelType.TEXT_TO_SPEECH
        else CONFIDENCE_STRONG,
        evidence=f"{family} voice model" if family else "declared a speech synthesiser",
    )


def _asr_model(profile: AssetProfile) -> Classification | None:
    """Claim speech recognition models."""
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    family = family_of(profile, ASR_FAMILIES)
    marker = profile.matches(("speech-recognition", "automatic-speech", "asr", "stt"))

    if declared != ModelType.SPEECH_RECOGNITION and family is None and marker is None:
        return None

    return Classification(
        category="speech", task="speech_recognition", domain="speech", family=family,
        modalities=("audio", "text"),
        confidence=CONFIDENCE_CERTAIN if declared == ModelType.SPEECH_RECOGNITION
        else CONFIDENCE_STRONG,
        evidence=f"{family} recogniser" if family else "declared a speech recogniser",
    )


def _audio_model(profile: AssetProfile) -> Classification | None:
    """Claim audio models that neither transcribe nor speak."""
    if not is_model(profile):
        return None
    if (profile.model.model_type if profile.model else None) != ModelType.AUDIO:
        return None

    haystack = profile.haystack
    if "separation" in haystack or "demucs" in haystack:
        task = "source_separation"
    elif "emotion" in haystack:
        task = "emotion_recognition"
    elif "speaker" in haystack or "diariz" in haystack:
        task = "speaker_recognition"
    else:
        task = "audio_classification"

    return Classification(
        category="audio", task=task, domain="audio", modalities=("audio",),
        confidence=CONFIDENCE_STRONG, evidence="declared an audio model",
    )


def _audio_dataset(profile: AssetProfile) -> Classification | None:
    """Claim datasets whose samples are sound."""
    if not is_dataset(profile):
        return None

    family = family_of(profile, AUDIO_DATASET_FAMILIES)
    clips = audio_count(profile)
    layout = declared_format(profile)

    if layout != DatasetFormat.AUDIO and family is None and clips < 1:
        return None

    haystack = profile.haystack
    if "speaker" in haystack or "voxceleb" in haystack:
        task = "speaker_recognition"
    elif family in ("AudioSet", "ESC-50", "UrbanSound") or "classification" in haystack:
        task = "audio_classification"
    elif family == "LJSpeech" or "tts" in haystack:
        task = "speech_synthesis"
    else:
        task = "speech_recognition"

    return Classification(
        category="audio_dataset", task=task, domain="audio", family=family,
        modalities=("audio",), confidence=CONFIDENCE_STRONG,
        evidence=f"{family} corpus" if family else f"{clips:,} audio file(s)",
    )
