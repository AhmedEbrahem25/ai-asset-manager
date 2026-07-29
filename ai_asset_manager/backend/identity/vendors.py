r"""Who ships models, where they put them, and what the words in between mean.

Three tables and nothing else. They are long because the knowledge is long — there is no
rule that derives "``screen_ai`` means optical character recognition" — but they are only
data, and adding a vendor is adding a tuple.

:data:`VENDORS` is matched against path *segments*, most specific first. That ordering is
load-bearing: ``Microsoft VS Code`` must be tried before ``Microsoft``, or every VS Code
model would be attributed to the parent company and lose the product that actually ships it.
"""

from __future__ import annotations

#: ``(segment markers, vendor, product, source id)``.
#:
#: A marker matches when it equals a path segment or appears in one, compared lower-cased.
#: ``source`` is the short identifier the catalogue records — the answer to "which piece of
#: software put this here?" — and is deliberately coarser than the product name.
#:
#: Ordered most specific first. Entries naming a product must precede the bare vendor.
VENDORS: tuple[tuple[tuple[str, ...], str, str | None, str], ...] = (
    # -- AI tooling and model stores ---------------------------------------
    (("huggingface", "hf_home", "hf_hub"), "HuggingFace", "Hub Cache", "huggingface"),
    ((".ollama", "ollama"), "Ollama", "Ollama", "ollama"),
    (("lm-studio", "lmstudio", "lm studio"), "LM Studio", "LM Studio", "lm_studio"),
    (("comfyui", "comfy"), "ComfyUI", "ComfyUI", "comfyui"),
    (("stable-diffusion-webui", "automatic1111", "sd-webui"),
     "AUTOMATIC1111", "Stable Diffusion WebUI", "sd_webui"),
    (("text-generation-webui", "oobabooga"), "oobabooga", "Text Generation WebUI", "textgen"),
    (("gpt4all",), "Nomic", "GPT4All", "gpt4all"),
    (("jan", "janai"), "Jan", "Jan", "jan"),
    (("koboldcpp", "kobold"), "KoboldCpp", "KoboldCpp", "koboldcpp"),
    (("llama.cpp", "llamacpp"), "llama.cpp", "llama.cpp", "llama_cpp"),
    (("torch\\hub", "torch/hub", "torch_hub"), "PyTorch", "PyTorch Hub", "pytorch_hub"),
    (("timm",), "timm", "timm", "timm"),
    (("ultralytics",), "Ultralytics", "Ultralytics", "ultralytics"),
    (("faster-whisper", "faster_whisper"), "SYSTRAN", "faster-whisper", "faster_whisper"),
    (("whisper",), "OpenAI", "Whisper", "whisper"),
    (("paddleocr", "paddlex", "paddle"), "PaddlePaddle", "PaddleOCR", "paddleocr"),
    (("easyocr",), "JaidedAI", "EasyOCR", "easyocr"),
    (("kraken",), "Kraken", "Kraken OCR", "kraken"),
    (("tesseract", "tessdata"), "Tesseract", "Tesseract OCR", "tesseract"),
    (("datalab", "surya"), "Datalab", "Surya OCR", "surya"),
    (("doctr",), "Mindee", "docTR", "doctr"),
    (("insightface",), "InsightFace", "InsightFace", "insightface"),
    (("deepface",), "DeepFace", "DeepFace", "deepface"),
    (("u2net", "rembg"), "rembg", "rembg", "rembg"),
    (("mediapipe",), "Google", "MediaPipe", "mediapipe"),
    (("spacy",), "Explosion", "spaCy", "spacy"),
    (("nltk_data", "nltk"), "NLTK", "NLTK", "nltk"),
    (("gensim-data", "gensim"), "Gensim", "Gensim", "gensim"),
    (("sentence_transformers", "sentence-transformers"),
     "Sentence-Transformers", "Sentence-Transformers", "sentence_transformers"),
    (("openvino",), "Intel", "OpenVINO", "openvino"),
    # ONNX and TFLite are deliberately absent. They are file formats, not sources: a model
    # is not "from ONNX" any more than a document is from PDF, and matching them here made
    # every `model.onnx` on the machine appear to be shipped by the runtime that reads it.

    # -- Browsers -----------------------------------------------------------
    (("google\\chrome", "google/chrome", "chrome"), "Google", "Chrome", "chrome"),
    (("microsoft\\edge", "microsoft/edge", "msedge"), "Microsoft", "Edge", "edge"),
    (("brave-browser", "bravesoftware", "brave"), "Brave", "Brave", "brave"),
    (("mozilla", "firefox"), "Mozilla", "Firefox", "firefox"),
    (("opera software", "opera"), "Opera", "Opera", "opera"),
    (("vivaldi",), "Vivaldi", "Vivaldi", "vivaldi"),
    (("chromium",), "Chromium", "Chromium", "chromium"),

    # -- Editors and developer tools ---------------------------------------
    (("microsoft vs code", "vscode", "code - insiders", ".vscode"),
     "Microsoft", "VS Code", "vscode"),
    (("cursor",), "Anysphere", "Cursor", "cursor"),
    ((".antigravity", "antigravity"), "Google", "Antigravity", "antigravity"),
    (("windsurf",), "Codeium", "Windsurf", "windsurf"),
    (("jetbrains", "intellij", "pycharm", "webstorm"), "JetBrains", "JetBrains IDE",
     "jetbrains"),
    (("sublime text", "sublime"), "Sublime HQ", "Sublime Text", "sublime"),
    (("github copilot", "copilot"), "GitHub", "Copilot", "copilot"),
    (("postman",), "Postman", "Postman", "postman"),
    (("docker",), "Docker", "Docker", "docker"),

    # -- Productivity and communication ------------------------------------
    (("microsoft office", "office16", "office15", "\\office\\", "/office/"),
     "Microsoft", "Office", "office"),
    (("onenote",), "Microsoft", "OneNote", "onenote"),
    (("microsoft teams", "teams"), "Microsoft", "Teams", "teams"),
    (("windows photos", "microsoft.windows.photos"), "Microsoft", "Photos", "photos"),
    (("powertoys",), "Microsoft", "PowerToys", "powertoys"),
    (("zoom",), "Zoom", "Zoom", "zoom"),
    (("webex",), "Cisco", "Webex", "webex"),
    (("slack",), "Slack", "Slack", "slack"),
    (("discord",), "Discord", "Discord", "discord"),
    (("telegram",), "Telegram", "Telegram", "telegram"),
    (("whatsapp",), "WhatsApp", "WhatsApp", "whatsapp"),
    (("grammarly",), "Grammarly", "Grammarly", "grammarly"),
    (("notion",), "Notion", "Notion", "notion"),
    (("obs-studio", "obs"), "OBS Project", "OBS Studio", "obs"),
    (("spotify",), "Spotify", "Spotify", "spotify"),
    (("dropbox",), "Dropbox", "Dropbox", "dropbox"),

    # -- Creative and imaging ----------------------------------------------
    (("adobe",), "Adobe", "Creative Cloud", "adobe"),
    (("topaz labs", "topaz"), "Topaz Labs", "Topaz", "topaz"),
    (("davinci resolve", "blackmagic"), "Blackmagic", "DaVinci Resolve", "resolve"),
    (("capcut",), "ByteDance", "CapCut", "capcut"),
    (("krita",), "Krita", "Krita", "krita"),
    (("gimp",), "GIMP", "GIMP", "gimp"),
    (("blender",), "Blender", "Blender", "blender"),

    # -- Vendors and platform components -----------------------------------
    (("nvidia", "nvidia corporation"), "NVIDIA", None, "nvidia"),
    (("intel", "intel corporation"), "Intel", None, "intel"),
    (("realsense",), "Intel", "RealSense", "realsense"),
    # iVCam is e2eSoft's phone-as-webcam app. It was briefly attributed to Intel here on
    # the strength of the letters "cam", which is the kind of guess a vendor table exists
    # to stop rather than to make.
    (("ivcam",), "e2eSoft", "iVCam", "ivcam"),
    (("amd", "radeon"), "AMD", None, "amd"),
    (("qualcomm", "snapdragon"), "Qualcomm", None, "qualcomm"),
    (("logitech", "logi"), "Logitech", None, "logitech"),
    (("elgato",), "Elgato", None, "elgato"),
    (("windowsapps", "systemapps", "windows.ai"), "Microsoft", "Windows", "windows"),
    (("google",), "Google", None, "google"),
    (("microsoft",), "Microsoft", None, "microsoft"),
    (("apple",), "Apple", None, "apple"),
    (("samsung",), "Samsung", None, "samsung"),
    (("gemini",), "Google", "Gemini", "gemini"),
)

#: ``(component markers, component label, task label)``.
#:
#: Matched against the path segments below the product and against the filename. The task
#: is what the component *does*, and is left ``None`` where the component name says what
#: something is without saying what it is for.
#:
#: Ordered longest-marker-first within each concern, because these are substring matches:
#: ``text_recognition`` has to be tried before ``recognition``.
COMPONENTS: tuple[tuple[tuple[str, ...], str, str | None], ...] = (
    # -- document and text understanding -----------------------------------
    (("screen_ai", "screenai", "screen-ai"), "ScreenAI", "OCR"),
    (("text_recognition", "text-recognition", "rec_infer", "_rec_", "recognition"),
     "Text Recognition", "OCR"),
    (("text_detection", "text-detection", "det_infer", "_det_", "detection_model"),
     "Text Detection", "OCR"),
    (("layout_analysis", "layout_detect", "layout"), "Layout Analysis", "OCR"),
    (("table_rec", "table_structure", "table"), "Table Recognition", "OCR"),
    (("reading_order", "order_detection"), "Reading Order", "OCR"),
    (("handwriting", "htr"), "Handwriting Recognition", "OCR"),
    (("ocr", "tessdata"), "OCR", "OCR"),
    (("document_ai", "docai", "docparse"), "Document Understanding", "OCR"),
    (("barcode", "qrcode", "qr_code"), "Barcode Reader", None),

    # -- speech and audio ---------------------------------------------------
    (("speech_recognition", "speech-recognition", "asr", "stt", "transcription"),
     "Speech Recognition", "Speech Recognition"),
    (("rnnt", "conformer", "wav2vec", "wav2letter"), "Speech Encoder", "Speech Recognition"),
    (("text_to_speech", "text-to-speech", "tts", "vocoder", "piper", "vits"),
     "Speech Synthesis", "Text to Speech"),
    (("noise_suppression", "noise-suppression", "denoise", "rnnoise", "krisp", "nsnet"),
     "Noise Suppression", "Audio Enhancement"),
    (("echo_cancel", "aec"), "Echo Cancellation", "Audio Enhancement"),
    (("voice_activity", "vad_", "_vad", "silero"), "Voice Activity Detection", "Audio"),
    (("speaker_id", "speaker-id", "diariz", "voiceprint"), "Speaker Recognition", "Audio"),
    (("keyword_spotting", "wake_word", "wakeword", "hotword"), "Wake Word", "Audio"),
    (("language_id", "language_detection", "lang_id", "lid"), "Language Identification",
     "Speech Recognition"),

    # -- vision -------------------------------------------------------------
    (("virtual_background", "background_blur", "background_seg", "selfie_seg", "matting"),
     "Background Segmentation", "Segmentation"),
    (("background_removal", "rembg", "u2net"), "Background Removal", "Segmentation"),
    (("face_detection", "face_detect", "facedetect", "blazeface", "retinaface", "scrfd"),
     "Face Detection", "Object Detection"),
    (("face_recognition", "facerec", "arcface", "facenet", "face_embed"),
     "Face Recognition", "Embedding"),
    (("face_landmark", "landmark", "facemesh", "face_mesh"), "Face Landmarks", "Pose"),
    (("hand_landmark", "hand_track", "gesture"), "Hand Tracking", "Pose"),
    (("pose_estimation", "pose_landmark", "posenet", "movenet", "openpose"),
     "Pose Estimation", "Pose"),
    (("gaze", "eye_track", "eyetrack"), "Gaze Estimation", "Pose"),
    (("object_detection", "obj_detect", "yolo", "ssd_mobilenet", "efficientdet"),
     "Object Detection", "Object Detection"),
    (("segmentation", "segment_anything", "deeplab", "unet"), "Segmentation", "Segmentation"),
    (("super_resolution", "superres", "upscale", "esrgan", "waifu"),
     "Super Resolution", "Image Enhancement"),
    (("depth_estimation", "depth_anything", "midas", "monodepth"), "Depth Estimation",
     "Depth Estimation"),
    (("image_classification", "imagenet", "mobilenet", "efficientnet", "convnext"),
     "Image Classification", "Image Classification"),
    (("image_embedding", "image_embed", "clip"), "Image Embedding", "Embedding"),
    (("nsfw", "content_safety", "safety_checker", "moderation"), "Content Safety", None),
    (("scene_detect", "shot_detect"), "Scene Detection", None),
    (("photo_enhance", "hdr", "denoiser"), "Photo Enhancement", "Image Enhancement"),

    # -- language and assistance -------------------------------------------
    (("optimization_guide", "optimizationguide", "optimization-guide"),
     "Optimization Guide", None),
    (("intelliphp", "intellicode", "intellisense", "code_completion", "autocomplete",
      "completion_model"), "Code Completion", "Code Completion"),
    (("word_prediction", "wordprediction", "next_word", "text_prediction", "predictive",
      "augloop"), "Text Prediction", "Text Generation"),
    (("spell", "autocorrect", "grammar", "proofing"), "Proofing", None),
    (("translation", "translate", "opus-mt", "nmt", "marian"), "Translation", "Translation"),
    (("summariz", "summaris"), "Summarisation", "Summarisation"),
    (("sentiment",), "Sentiment Analysis", "Sentiment Analysis"),
    (("intent", "slot_filling"), "Intent Classification", "Classification"),
    (("embedding", "embed_model", "sentence_embed", "text_embed"), "Embedding", "Embedding"),
    (("rerank",), "Reranker", "Reranking"),
    (("entity", "ner_"), "Entity Recognition", "Classification"),
    (("safe_browsing", "safebrowsing", "phishing"), "Safe Browsing", None),
    (("spam", "junk_filter"), "Spam Filter", "Classification"),
    (("smart_reply", "smartreply", "suggested_replies"), "Smart Reply", "Text Generation"),
    (("search_ranking", "ranking_model", "ranker"), "Ranking", "Reranking"),
    (("recommend",), "Recommendation", None),
    (("anomaly",), "Anomaly Detection", None),
    (("telemetry_model", "crash_predict"), "Telemetry", None),
)

#: Path segments that carry no information about what a model is. Filtered out before a
#: component is chosen, so ``Chrome\\User Data\\Default\\screen_ai\\1.0.3`` yields
#: ``screen_ai`` rather than ``1.0.3``.
NOISE_SEGMENTS: frozenset[str] = frozenset(
    {
        "appdata", "local", "locallow", "roaming", "users", "user data", "userdata",
        "default", "profile", "profile 1", "program files", "program files (x86)",
        "programdata", "windows", "system32", "syswow64", "bin", "bin64", "lib", "libs",
        "res", "resources", "cloudresources", "modelresources", "assets", "content",
        "contents", "app", "application", "server",
        "apps", "cache", "caches", ".cache", "cached", "tmp", "temp", "data", "db",
        "store", "storage", "files", "file", "current", "latest", "stable", "release",
        "build", "builds", "dist", "out", "output", "install", "installed", "setup",
        "models", "model", "ml", "ai", "mlmodels", "ml_models", "weights", "checkpoints",
        "checkpoint", "artifacts", "hub", "snapshots", "blobs", "refs", "manifests",
        "registry", "library", "packages", "package", "extensions", "extension",
        "plugins", "plugin", "modules", "node_modules", "site-packages", "python",
        "win32", "win64", "x64", "x86", "amd64", "arm64", "cpu", "gpu", "cuda",
        "onnx", "tflite", "tf", "torch", "pytorch", "saved_model", "variables",
        "assets.extra", "1", "2", "3", "v1", "v2", "v3",
    }
)

#: Words that make a *filename* uninformative. A file called any of these is named after
#: nothing, which is the whole problem this package exists to solve.
GENERIC_NAMES: frozenset[str] = frozenset(
    {
        "model", "models", "weights", "weight", "net", "network", "graph", "frozen",
        "frozen_graph", "saved_model", "checkpoint", "ckpt", "final", "best", "last",
        "output", "export", "exported", "converted", "quantized", "quantised",
        "optimized", "optimised", "pytorch_model", "tf_model", "flax_model",
        "model_quantized", "model_optimized", "model_float32", "model_float16",
        "model_int8", "model_fp16", "model_fp32", "model_q4", "model_q8", "data",
        "default", "main", "base", "common", "shared", "generic", "unnamed", "untitled",
        "temp", "tmp", "test", "sample", "example", "demo", "new", "copy", "backup",
        "inference", "predictor", "classifier", "encoder", "decoder", "embedding",
    }
)

#: Acronyms and product names whose casing must survive title-casing.
CASING: dict[str, str] = {
    "ocr": "OCR", "asr": "ASR", "tts": "TTS", "stt": "STT", "nlp": "NLP", "ai": "AI",
    "ml": "ML", "vad": "VAD", "lid": "LID", "id": "ID", "ui": "UI", "api": "API",
    "gpu": "GPU", "cpu": "CPU", "npu": "NPU", "llm": "LLM", "vlm": "VLM",
    "rnnt": "RNN-T", "lstm": "LSTM", "gru": "GRU", "cnn": "CNN", "rnn": "RNN",
    "bert": "BERT", "gpt": "GPT", "vit": "ViT", "clip": "CLIP", "yolo": "YOLO",
    "sam": "SAM", "svm": "SVM", "knn": "KNN", "hdr": "HDR", "qr": "QR",
    "aec": "AEC", "ner": "NER", "ids": "IDS", "iot": "IoT", "pdf": "PDF",
    "vs": "VS", "sd": "SD", "sdxl": "SDXL", "tf": "TF", "onnx": "ONNX",
    "3d": "3D", "2d": "2D", "us": "US", "uk": "UK",
}
