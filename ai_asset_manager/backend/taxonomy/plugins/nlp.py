"""Language models, embeddings and text datasets.

Deliberately low in the priority order. Almost every modern model is a transformer with a
language head somewhere in it, so a language rule that ran early would claim OCR engines,
vision-language models and speech decoders alike. By the time these rules run, the plugins
that can prove a more specific purpose have already declined.
"""

from __future__ import annotations

from ai_asset_manager.backend.models.enums import DatasetFormat, Framework, ModelType
from ai_asset_manager.backend.taxonomy.plugins._shared import (
    TABULAR_EXTENSIONS,
    declared_format,
    family_of,
    is_dataset,
    is_model,
)
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_STRONG,
    CONFIDENCE_WEAK,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: Generative language model families.
LLM_FAMILIES = (
    ("Llama", ("llama", "tinyllama", "codellama")),
    ("Qwen", ("qwen",)),
    ("Mixtral", ("mixtral",)),
    ("Mistral", ("mistral", "ministral")),
    ("Phi", ("phi-", "phi_", "phi3", "phi4")),
    ("Gemma", ("gemma",)),
    ("DeepSeek", ("deepseek",)),
    ("Falcon", ("falcon",)),
    ("Yi", ("yi-1.5", "yi-6b", "yi-34b")),
    ("Command R", ("command-r", "c4ai-command")),
    ("Granite", ("granite",)),
    ("SmolLM", ("smollm",)),
    ("StableLM", ("stablelm",)),
    ("OLMo", ("olmo",)),
    ("Nemotron", ("nemotron",)),
    ("GLM", ("chatglm", "glm-4", "glm4")),
    ("Kimi", ("kimi",)),
    ("MiniMax", ("minimax",)),
    ("MPT", ("mpt-",)),
    ("GPT", ("gpt-oss", "gpt2", "gpt-neo", "gpt-j", "gptq-")),
    ("Aya", ("aya-",)),
    ("Jais", ("jais",)),
    ("AceGPT", ("acegpt",)),
)

#: Encoder families, used for embeddings, classification and token tagging rather than
#: generation.
ENCODER_FAMILIES = (
    ("BGE", ("bge-", "bge_")),
    ("GTE", ("gte-", "gte_")),
    ("E5", ("e5-", "multilingual-e5")),
    ("Nomic", ("nomic-embed", "nomic_embed")),
    ("Jina", ("jina-embed", "jina-reranker", "jina-bert")),
    ("Stella", ("stella_en", "stella-en")),
    ("MiniLM", ("minilm",)),
    ("MPNet", ("mpnet",)),
    ("SimCSE", ("simcse",)),
    ("DeBERTa", ("deberta",)),
    ("DistilBERT", ("distilbert",)),
    ("ALBERT", ("albert",)),
    ("ELECTRA", ("electra",)),
    ("RoBERTa", ("roberta", "camembert")),
    ("XLM-R", ("xlm-roberta", "xlm_roberta")),
    ("MARBERT", ("marbert",)),
    ("AraBERT", ("arabert",)),
    ("BERT", ("bert",)),
    ("T5", ("byt5", "mt5", "flan-t5", "t5-")),
    ("BART", ("bart",)),
)

#: Text corpora people download by name.
NLP_DATASET_FAMILIES = (
    ("Wikipedia", ("wikipedia", "wikitext")),
    ("C4", ("c4-", "allenai/c4")),
    ("The Pile", ("the-pile", "the_pile", "eleutherai/pile")),
    ("OSCAR", ("oscar",)),
    ("RedPajama", ("redpajama",)),
    ("FineWeb", ("fineweb",)),
    ("Dolma", ("dolma",)),
    ("SQuAD", ("squad",)),
    ("GLUE", ("glue",)),
    ("GSM8K", ("gsm8k",)),
    ("MMLU", ("mmlu",)),
    ("Alpaca", ("alpaca",)),
    ("ShareGPT", ("sharegpt",)),
    ("UltraChat", ("ultrachat",)),
    ("OpenOrca", ("openorca", "open-orca")),
    ("Dolly", ("databricks-dolly", "dolly-15k")),
    ("IMDB", ("imdb",)),
    ("CoNLL", ("conll",)),
)

#: Transformer heads that predict a label over text. The one reliable way to tell a
#: sentiment model from an image classifier when both are recorded as "classification".
_LABEL_HEADS = ("forsequenceclassification", "fortokenclassification",
                "formultiplechoice", "forquestionanswering")

TASKS = (
    Task(id="chat", label="Chat", domain="nlp", order=10),
    Task(id="text_generation", label="Text Generation", domain="nlp", order=20),
    Task(id="instruction_tuning", label="Instruction Tuning", domain="nlp", order=30),
    Task(id="embeddings", label="Embeddings", domain="nlp", order=40),
    Task(id="reranking", label="Reranking", domain="nlp", order=50),
    Task(id="rag", label="Retrieval-Augmented Generation", domain="nlp", order=60),
    Task(id="text_classification", label="Text Classification", domain="nlp", order=70),
    Task(id="sentiment_analysis", label="Sentiment Analysis", domain="nlp", order=80),
    Task(id="named_entity_recognition", label="Named Entity Recognition",
         domain="nlp", order=90),
    Task(id="question_answering", label="Question Answering", domain="nlp", order=100),
    Task(id="summarization", label="Summarization", domain="nlp", order=110),
    Task(id="machine_translation", label="Machine Translation", domain="nlp", order=120),
    Task(id="fill_mask", label="Masked Language Modelling", domain="nlp", order=130),
    Task(id="language_modelling", label="Language Modelling", domain="nlp", order=140),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register language categories, tasks and classifiers."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="llm", label="LLM", section="models", order=10, domain="nlp",
                 aliases=("llms", "language-models"),
                 description="Generative language models.")
    )
    registry.add_category(
        Category(id="embedding", label="Embedding", section="models", order=100,
                 domain="nlp", aliases=("embeddings",))
    )
    registry.add_category(
        Category(id="reranker", label="Reranker", section="models", order=110, domain="nlp",
                 aliases=("rerankers", "cross-encoders"))
    )
    registry.add_category(
        Category(id="text_classification", label="Text Classification", section="models",
                 order=105, domain="nlp",
                 aliases=("text-classifiers", "sentiment", "classifiers-text"),
                 description="Encoders with a label head: sentiment, topic, NER.")
    )
    registry.add_category(
        Category(id="nlp_dataset", label="NLP Dataset", section="datasets", order=270,
                 domain="nlp", aliases=("nlp-datasets", "text-datasets"))
    )

    # Retrieval work needs both halves, and asking for one nearly always means both.
    registry.add_alias("retrieval", ("embedding", "reranker"))
    registry.add_alias("embeddings", ("embedding", "reranker"))

    registry.add_classifier(_reranker_model, name="nlp.reranker", priority=330)
    registry.add_classifier(_text_classifier, name="nlp.text-classifier", priority=325)
    registry.add_classifier(_embedding_model, name="nlp.embedding", priority=320)
    registry.add_classifier(_language_model, name="nlp.llm", priority=310)
    registry.add_classifier(_nlp_dataset, name="nlp.dataset", priority=300)


def _reranker_model(profile: AssetProfile) -> Classification | None:
    """Claim cross-encoder rerankers.

    Ahead of embeddings because a reranker is an encoder too, and its name usually says
    ``reranker`` while its architecture says ``ForSequenceClassification``.
    """
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    marker = profile.matches(("reranker", "rerank", "cross-encoder", "crossencoder"))
    if declared != ModelType.RERANKER and marker is None:
        return None

    return Classification(
        category="reranker", task="reranking", domain="nlp",
        family=family_of(profile, ENCODER_FAMILIES), modalities=("text",),
        confidence=CONFIDENCE_CERTAIN if declared == ModelType.RERANKER else CONFIDENCE_STRONG,
        evidence="declared a reranker" if declared == ModelType.RERANKER
        else f"name contains {marker!r}",
    )


def _text_classifier(profile: AssetProfile) -> Classification | None:
    """Claim encoders that predict a label rather than a vector.

    These are the models the catalogue is most likely to have mislabelled, because
    "classification" reads as a vision word and the scanner records vision classifiers and
    sentiment models under the same type. The head settles it: nothing puts
    ``ForSequenceClassification`` on a ResNet.
    """
    if not is_model(profile):
        return None

    architecture = (profile.model.architecture or "").lower() if profile.model else ""
    if not architecture.endswith(_LABEL_HEADS):
        return None

    haystack = profile.haystack
    if "ner" in haystack or architecture.endswith("fortokenclassification"):
        task = "named_entity_recognition"
    elif architecture.endswith("forquestionanswering"):
        task = "question_answering"
    elif "sentiment" in haystack or "emotion" in haystack or "polarity" in haystack:
        task = "sentiment_analysis"
    else:
        task = "text_classification"

    return Classification(
        category="text_classification", task=task, domain="nlp",
        family=family_of(profile, ENCODER_FAMILIES), modalities=("text",),
        confidence=CONFIDENCE_STRONG, evidence=f"{architecture} head",
    )


def _embedding_model(profile: AssetProfile) -> Classification | None:
    """Claim sentence embedders and the encoders that serve as them.

    Sentence-Transformers repositories are unambiguous. Bare encoder checkpoints are not:
    a ``BertForPreTraining`` checkpoint has no declared purpose, but what anyone actually
    does with one is produce embeddings, so that is where the inventory files it.
    """
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    architecture = (profile.model.architecture or "").lower() if profile.model else ""
    family = family_of(profile, ENCODER_FAMILIES)

    if declared == ModelType.EMBEDDING:
        evidence, confidence = "declared an embedding model", CONFIDENCE_CERTAIN
    elif profile.framework == Framework.SENTENCE_TRANSFORMERS:
        evidence, confidence = "sentence-transformers repository", CONFIDENCE_CERTAIN
    elif profile.matches(("embedding", "-embed", "_embed", "sentence-transformer")):
        evidence, confidence = "named as an embedding model", CONFIDENCE_STRONG
    elif architecture and (
        architecture.endswith(("model", "formaskedlm", "forpretraining"))
        or "sentencetransformer" in architecture
    ):
        evidence, confidence = f"{architecture} encoder", CONFIDENCE_WEAK
    elif family is not None and declared in (None, ModelType.UNKNOWN):
        evidence, confidence = f"{family} encoder", CONFIDENCE_WEAK
    else:
        return None

    return Classification(
        category="embedding", task="embeddings", domain="nlp", family=family,
        modalities=("text",), confidence=confidence, evidence=evidence,
    )


def _language_model(profile: AssetProfile) -> Classification | None:
    """Claim generative language models."""
    if not is_model(profile):
        return None

    declared = profile.model.model_type if profile.model else None
    architecture = (profile.model.architecture or "").lower() if profile.model else ""
    family = family_of(profile, LLM_FAMILIES)

    if declared == ModelType.LLM:
        evidence, confidence = "declared a language model", CONFIDENCE_CERTAIN
    elif architecture.endswith(("forcausallm", "forconditionalgeneration", "lmheadmodel")):
        evidence, confidence = f"{architecture} head", CONFIDENCE_STRONG
    elif family is not None:
        evidence, confidence = f"{family} family", CONFIDENCE_WEAK
    else:
        return None

    haystack = profile.haystack
    if "instruct" in haystack or "chat" in haystack or "-it" in haystack:
        task = "chat"
    elif "base" in haystack:
        task = "language_modelling"
    else:
        task = "text_generation"

    return Classification(
        category="llm", task=task, domain="nlp", family=family, modalities=("text",),
        confidence=confidence, evidence=evidence,
    )


def _nlp_dataset(profile: AssetProfile) -> Classification | None:
    """Claim text corpora, including HuggingFace datasets that are just Parquet."""
    if not is_dataset(profile):
        return None

    layout = declared_format(profile)
    details = profile.dataset
    textual = layout in (DatasetFormat.NLP, DatasetFormat.HF_DATASET, DatasetFormat.TABULAR)

    if not textual:
        # A HuggingFace dataset with no images, videos or audio is text by elimination.
        has_media = bool(
            details and (details.num_images or details.num_videos or details.num_audio_files)
        )
        if has_media or not profile.files.count(*TABULAR_EXTENSIONS):
            return None

    haystack = profile.haystack
    if "instruct" in haystack or "alpaca" in haystack or "sharegpt" in haystack:
        task = "instruction_tuning"
    elif "squad" in haystack or "qa" in haystack:
        task = "question_answering"
    elif "sentiment" in haystack:
        task = "sentiment_analysis"
    elif "ner" in haystack or "conll" in haystack:
        task = "named_entity_recognition"
    elif "translation" in haystack:
        task = "machine_translation"
    elif "summar" in haystack:
        task = "summarization"
    else:
        task = "language_modelling"

    return Classification(
        category="nlp_dataset", task=task, domain="nlp",
        family=family_of(profile, NLP_DATASET_FAMILIES), modalities=("text",),
        confidence=CONFIDENCE_STRONG if textual else CONFIDENCE_WEAK,
        evidence=f"{layout} layout" if layout else "record files with no media",
    )
