"""Parsers for the HuggingFace JSON configuration family.

Covers ``config.json``, ``tokenizer_config.json``, ``generation_config.json``,
``model_index.json`` (Diffusers) and ``adapter_config.json`` (PEFT). These are the
highest-precedence source of truth: when a config states an architecture, nothing derived
from a filename gets to overrule it.
"""

from __future__ import annotations

from typing import Any

from ai_asset_manager.backend.models.enums import FactSource, Framework, ModelType, Precision
from ai_asset_manager.backend.parsers.base import BaseParser, FactSet
from ai_asset_manager.backend.scanner.context import DirectoryContext

#: Substrings of a ``config.json`` architecture entry mapped to a model type. Checked in
#: order, so more specific patterns must come first: ``LlavaForConditionalGeneration``
#: must match vision-language before the generic ``ForCausalLM`` rule sees it.
ARCHITECTURE_PATTERNS: tuple[tuple[tuple[str, ...], ModelType], ...] = (
    (("llava", "qwen2vl", "qwen2_vl", "qwen2_5_vl", "idefics", "paligemma", "instructblip",
      "internvl", "minicpmv", "pixtral", "fuyu", "kosmos", "mllama", "smolvlm", "molmo",
      "videollama", "cogvlm", "glm4v", "aria"), ModelType.VISION_LANGUAGE),
    # OCR must precede the generic causal-LM rule: modern OCR models are named
    # `...OCRForCausalLM` / `...OCRForConditionalGeneration`, so the LLM rule would
    # otherwise swallow every one of them.
    (("trocr", "donut", "nougat", "paddleocr", "ocr"), ModelType.OCR),
    # The bare names in the second half are GGUF `general.architecture` values, which are
    # lower-case family names rather than transformers class names.
    (("forcausallm", "forconditionalgeneration", "lmheadmodel", "gptneox", "gptj",
      "mistral", "mixtral", "falcon", "mpt", "phi", "gemma", "olmo", "starcoder",
      "codegen", "stablelm", "baichuan", "internlm", "yi", "deepseek", "granite",
      "cohere", "dbrx", "jamba", "recurrentgemma", "nemotron", "exaone",
      "llama", "qwen", "gpt2", "gptneo", "bloom", "opt", "chatglm", "glm", "rwkv",
      "mamba", "command-r", "smollm", "minicpm", "orion", "xverse", "plamo"),
     ModelType.LLM),
    (("whisper", "wav2vec2", "hubert", "wavlm", "seamless", "speech2text", "unispeech",
      "sew", "data2vecaudio", "moonshine"), ModelType.SPEECH_RECOGNITION),
    (("speecht5", "bark", "vits", "musicgen", "fastspeech", "parler", "xtts", "encodec"),
     ModelType.TEXT_TO_SPEECH),
    (("clip", "siglip", "blip", "align", "altclip", "chineseclip", "owlvit", "owlv2",
      "groundingdino"), ModelType.MULTIMODAL),
    (("detr", "yolos", "deformabledetr", "conditionaldetr", "rtdetr", "dino",
      "fordetection", "objectdetection", "table_transformer"), ModelType.OBJECT_DETECTION),
    (("segformer", "maskformer", "mask2former", "upernet", "forsemanticsegmentation",
      "forinstancesegmentation", "forsegmentation", "sam", "oneformer", "beitsegment"),
     ModelType.SEGMENTATION),
    (("forsequenceclassification", "fortokenclassification", "forimageclassification",
      "vit", "swin", "convnext", "resnet", "efficientnet", "deit", "beit", "dinov2",
      "regnet", "mobilenet"), ModelType.CLASSIFICATION),
    # Encoder-only checkpoints. `ForPreTraining` heads are included because a saved
    # pretraining checkpoint is kept to be used as an encoder, not to keep pretraining.
    (("formaskedlm", "forpretraining", "bertmodel", "robertamodel", "xlmroberta",
      "distilbert", "electra", "deberta", "mpnet", "albert", "e5", "bge", "gte",
      "nomic", "jina", "sentencetransformer"),
     ModelType.EMBEDDING),
    (("unet2dcondition", "stablediffusion", "flux", "sd3", "pixart", "kandinsky",
      "latentdiffusion", "autoencoderkl", "controlnet", "dit"), ModelType.IMAGE_GENERATION),
    (("forposeestimation", "vitpose", "keypoint"), ModelType.POSE),
)

#: ``torch_dtype`` values mapped to a precision class.
TORCH_DTYPE_PRECISION: dict[str, Precision] = {
    "float32": Precision.FP32,
    "float": Precision.FP32,
    "float16": Precision.FP16,
    "half": Precision.FP16,
    "bfloat16": Precision.BF16,
    "float8_e4m3fn": Precision.FP8,
    "int8": Precision.INT8,
    "uint8": Precision.INT8,
}


def classify_architecture(*candidates: str | None) -> ModelType | None:
    """Infer a model type from architecture or model-type strings.

    Args:
        *candidates: Values such as ``"Qwen2ForCausalLM"`` or ``"llama"``. ``None`` values
            are skipped.

    Returns:
        The matching :class:`ModelType`, or ``None`` when nothing matches.

    Examples:
        >>> classify_architecture("Qwen2ForCausalLM")
        <ModelType.LLM: 'llm'>
        >>> classify_architecture("Qwen2VLForConditionalGeneration")
        <ModelType.VISION_LANGUAGE: 'vision_language'>
        >>> classify_architecture("WhisperForConditionalGeneration")
        <ModelType.SPEECH_RECOGNITION: 'speech_recognition'>
    """
    haystack = " ".join(value.lower() for value in candidates if value)
    if not haystack:
        return None
    for needles, model_type in ARCHITECTURE_PATTERNS:
        if any(needle in haystack for needle in needles):
            return model_type
    return None


def _first_str(value: Any) -> str | None:
    """Return a string from either a scalar or the head of a list."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    return None


class HfConfigParser(BaseParser):
    """Reads ``config.json``, the canonical description of a transformers model."""

    name = "hf_config"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether a readable ``config.json`` is present."""
        return isinstance(ctx.read_json("config.json"), dict)

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Extract architecture, dimensions and quantisation from ``config.json``."""
        facts = self._new_facts()
        config = ctx.read_json("config.json")
        if not isinstance(config, dict):
            return facts

        architecture = _first_str(config.get("architectures")) or config.get("model_type")
        facts.add("architecture", architecture, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("framework", Framework.TRANSFORMERS.value, source=FactSource.EXPLICIT_CONFIG,
                  confidence=0.7, origin=self.name)

        model_type = classify_architecture(
            _first_str(config.get("architectures")),
            config.get("model_type") if isinstance(config.get("model_type"), str) else None,
        )
        if model_type is not None:
            facts.add("model_type", model_type.value, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)

        for config_key, fact_key in (
            ("hidden_size", "hidden_size"),
            ("n_embd", "hidden_size"),
            ("d_model", "hidden_size"),
            ("num_hidden_layers", "num_layers"),
            ("n_layer", "num_layers"),
            ("num_layers", "num_layers"),
            ("vocab_size", "vocab_size"),
            ("max_position_embeddings", "context_length"),
            ("n_positions", "context_length"),
            ("max_seq_len", "context_length"),
        ):
            value = config.get(config_key)
            if isinstance(value, int) and value > 0:
                facts.add(fact_key, value, source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        dtype = config.get("torch_dtype") or config.get("dtype")
        if isinstance(dtype, str) and dtype.lower() in TORCH_DTYPE_PRECISION:
            facts.add("precision", TORCH_DTYPE_PRECISION[dtype.lower()].value,
                      source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        self._parse_quantization(facts, config)

        for key, target in (
            ("_name_or_path", "repo_id"),
            ("transformers_version", "transformers_version"),
            ("license", "license"),
        ):
            value = config.get(key)
            if isinstance(value, str) and value:
                facts.add(target, value, source=FactSource.EXPLICIT_CONFIG, confidence=0.6,
                          origin=self.name)

        return facts

    def _parse_quantization(self, facts: FactSet, config: dict[str, Any]) -> None:
        """Read a ``quantization_config`` block, as written by bitsandbytes, GPTQ and AWQ."""
        block = config.get("quantization_config")
        if not isinstance(block, dict):
            return

        method = block.get("quant_method")
        bits = block.get("bits") or block.get("w_bit")

        if block.get("load_in_4bit"):
            label = f"bnb-4bit-{block.get('bnb_4bit_quant_type', 'nf4')}"
            precision = Precision.INT4
        elif block.get("load_in_8bit"):
            label, precision = "bnb-8bit", Precision.INT8
        elif isinstance(method, str):
            label = f"{method}-{bits}bit" if isinstance(bits, int) else str(method)
            precision = Precision.INT4 if bits == 4 else Precision.INT8
        else:
            return

        facts.add("quantization", label, source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("precision", precision.value, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)


class AdapterConfigParser(BaseParser):
    """Reads ``adapter_config.json``, written by PEFT for LoRA and friends."""

    name = "adapter_config"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether a readable ``adapter_config.json`` is present."""
        return isinstance(ctx.read_json("adapter_config.json"), dict)

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Extract adapter rank, target modules and the base model it attaches to."""
        facts = self._new_facts()
        config = ctx.read_json("adapter_config.json")
        if not isinstance(config, dict):
            return facts

        facts.add("framework", Framework.PEFT.value, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("model_type", ModelType.LORA.value, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)

        peft_type = config.get("peft_type")
        if isinstance(peft_type, str):
            facts.add("adapter_type", peft_type, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)
            facts.add("architecture", peft_type, source=FactSource.EXPLICIT_CONFIG,
                      confidence=0.6, origin=self.name)

        base = config.get("base_model_name_or_path")
        if isinstance(base, str) and base:
            facts.add("base_model", base, source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        for key, target in (("r", "lora_rank"), ("lora_alpha", "lora_alpha")):
            value = config.get(key)
            if isinstance(value, int):
                facts.add(target, value, source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        targets = config.get("target_modules")
        if isinstance(targets, list):
            facts.add("target_modules", sorted(str(item) for item in targets),
                      source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        elif isinstance(targets, str):
            facts.add("target_modules", [targets], source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)

        task = config.get("task_type")
        if isinstance(task, str):
            facts.add("task", task, source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        return facts


class ModelIndexParser(BaseParser):
    """Reads ``model_index.json``, the Diffusers pipeline manifest."""

    name = "model_index"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether a readable ``model_index.json`` is present."""
        return isinstance(ctx.read_json("model_index.json"), dict)

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Extract the pipeline class and its component list."""
        facts = self._new_facts()
        index = ctx.read_json("model_index.json")
        if not isinstance(index, dict):
            return facts

        facts.add("framework", Framework.DIFFUSERS.value, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)
        facts.add("model_type", ModelType.IMAGE_GENERATION.value,
                  source=FactSource.EXPLICIT_CONFIG, confidence=0.9, origin=self.name)

        pipeline_class = index.get("_class_name")
        if isinstance(pipeline_class, str):
            facts.add("architecture", pipeline_class, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)
            facts.add("pipeline_class", pipeline_class, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)
            # Diffusers also ships audio and video pipelines under the same manifest.
            lowered = pipeline_class.lower()
            if "audio" in lowered or "music" in lowered:
                facts.add("model_type", ModelType.AUDIO.value,
                          source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        version = index.get("_diffusers_version")
        if isinstance(version, str):
            facts.add("diffusers_version", version, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)

        components = sorted(
            key for key, value in index.items()
            if not key.startswith("_") and isinstance(value, list)
        )
        if components:
            facts.add("components", components, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)

        return facts


class TokenizerConfigParser(BaseParser):
    """Reads ``tokenizer_config.json`` and ``generation_config.json``.

    Neither identifies a model on its own, but both carry corroborating detail — a chat
    template implies an instruction-tuned model, and the tokenizer class often names the
    architecture family when ``config.json`` is missing.
    """

    name = "tokenizer_config"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether either config file is present."""
        return ctx.has_any("tokenizer_config.json", "generation_config.json")

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Extract tokenizer class, chat-template presence and context length."""
        facts = self._new_facts()

        tokenizer = ctx.read_json("tokenizer_config.json")
        if isinstance(tokenizer, dict):
            facts.add("has_tokenizer", True, source=FactSource.EXPLICIT_CONFIG, origin=self.name)

            tokenizer_class = tokenizer.get("tokenizer_class")
            if isinstance(tokenizer_class, str):
                facts.add("tokenizer_class", tokenizer_class, source=FactSource.EXPLICIT_CONFIG,
                          origin=self.name)

            if tokenizer.get("chat_template"):
                facts.add("is_instruct", True, source=FactSource.EXPLICIT_CONFIG,
                          confidence=0.8, origin=self.name)

            max_length = tokenizer.get("model_max_length")
            # Tokenizers use a sentinel of 1e30 to mean "unbounded"; storing that as a
            # context length would put nonsense in the UI.
            if isinstance(max_length, int) and 0 < max_length < 10_000_000:
                facts.add("context_length", max_length, source=FactSource.EXPLICIT_CONFIG,
                          confidence=0.5, origin=self.name)

        generation = ctx.read_json("generation_config.json")
        if isinstance(generation, dict):
            facts.add("has_generation_config", True, source=FactSource.EXPLICIT_CONFIG,
                      origin=self.name)
            facts.add("model_type", ModelType.LLM.value, source=FactSource.EXPLICIT_CONFIG,
                      confidence=0.4, origin=self.name)

        return facts


class SentenceTransformerParser(BaseParser):
    """Reads the ``modules.json`` marker that identifies a Sentence-Transformers model."""

    name = "sentence_transformers"

    def supports(self, ctx: DirectoryContext) -> bool:
        """Report whether Sentence-Transformers marker files are present."""
        return ctx.has_any("modules.json", "config_sentence_transformers.json")

    def parse(self, ctx: DirectoryContext) -> FactSet:
        """Identify the model as an embedding model and read its pooling dimension."""
        facts = self._new_facts()
        modules = ctx.read_json("modules.json")
        st_config = ctx.read_json("config_sentence_transformers.json")

        if not isinstance(modules, list) and not isinstance(st_config, dict):
            return facts

        facts.add("framework", Framework.SENTENCE_TRANSFORMERS.value,
                  source=FactSource.EXPLICIT_CONFIG, origin=self.name)
        facts.add("model_type", ModelType.EMBEDDING.value, source=FactSource.EXPLICIT_CONFIG,
                  origin=self.name)

        pooling = ctx.child("1_Pooling")
        if pooling is not None:
            pooling_config = pooling.read_json("config.json")
            if isinstance(pooling_config, dict):
                dimension = pooling_config.get("word_embedding_dimension")
                if isinstance(dimension, int):
                    facts.add("embedding_dimension", dimension,
                              source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        if isinstance(st_config, dict):
            version = st_config.get("__version__")
            if isinstance(version, dict):
                st_version = version.get("sentence_transformers")
                if isinstance(st_version, str):
                    facts.add("sentence_transformers_version", st_version,
                              source=FactSource.EXPLICIT_CONFIG, origin=self.name)

        return facts
