# AI Asset Inventory - Full Machine Report

Generated 2026-07-29 10:07 UTC by [AI Asset Manager](../README.md) `aam`.

Every number below was read from the catalogue database. Producing this report walked no directory and opened no file, which is the same guarantee `aam inventory` makes: the tool is strictly read-only and can never move, rename or delete anything.

## At a glance

|  |  |
| --- | --- |
| **AI assets catalogued** | **154** |
| **Storage they occupy** | **41.7 GiB** |
| Files inside them | 1,929 |
| Average health | 99/100 |
| Needing attention | 4 |
| Relationships derived | 45 |
|  |  |
| Directories examined | 101,539 |
| Files examined | 700,435 |
| Bytes examined | 1.0 TiB |
| Scan time | 523.3s |

0.3% of the files examined turned out to belong to an AI asset. Separating that fraction from everything else is the whole job, and it is why the scanner classifies by directory *structure* rather than by name.

## What was scanned

| Root | Assets found | Last scanned |
| --- | --- | --- |
| `C:\` | 92 | 2026-07-29 10:04 |
| `F:\` | 62 | 2026-07-29 10:04 |

Pruned from the walk by default, in three groups:

- **Operating-system and vendor trees** - `Windows`, `Program Files`, `Program Files (x86)`, `ProgramData`, `$Recycle.Bin`, `System Volume Information`, `PerfLogs`, `Recovery`.
- **Browser, Electron and package-manager caches** - `Cache2`, `Code Cache`, `GPUCache`, `Service Worker`, `WebCache`, `INetCache`, `.npm`, `.nuget`, `.cargo`, `.rustup`, plus `AppData\Local\Temp`.
- **Build and dependency directories** - `node_modules`, `site-packages`, `dist-packages`, `.venv`, `venv`, `__pycache__`, `.git`, `build`, `target`, and the `uv` and `pip` wheel caches.

None of these has ever held a catalogued asset, and together they are the difference between a whole-drive scan that finishes and one that does not. Naming a pruned path explicitly still scans it -- the walker only checks the names of directories it *discovers*, never the root it was handed.

## What is here

| Section | Assets | Storage | Share of storage |
| --- | ---: | ---: | ---: |
| Models | 101 | 28.3 GiB | 67.9% |
| Experiments & Logs | 44 | 13.0 GiB | 31.2% |
| Datasets | 9 | 396.4 MiB | 0.9% |

### By category

| Category | Section | Assets | Storage | Largest single asset |
| --- | --- | ---: | ---: | --- |
| OCR Model | Models | 11 | 15.5 GiB | baidu/Unlimited-OCR |
| AI Project | Experiments & Logs | 4 | 8.4 GiB | thorn-nlp |
| LLM | Models | 4 | 6.9 GiB | deepseek-r1:8b |
| Checkpoint | Experiments & Logs | 10 | 4.6 GiB | best_dapt |
| Vision-Language | Models | 1 | 2.3 GiB | unsloth/qwen2-vl-2b-instruct-unsloth-b |
| Embedding | Models | 4 | 1.4 GiB | UBC-NLP/MARBERTv2 |
| Speech | Models | 6 | 817.2 MiB | large-v3-turbo |
| Text Classification | Models | 1 | 479.1 MiB | cardiffnlp/twitter-roberta-base-sentim |
| Model | Models | 51 | 377.4 MiB | model |
| NLP Dataset | Datasets | 4 | 195.5 MiB | criminal |
| Object Detection | Models | 3 | 170.4 MiB | fasterrcnn_resnet50_fpn_coco-258fb6c6 |
| Adapter / LoRA | Models | 1 | 125.7 MiB | NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Ins |
| Image Dataset | Datasets | 2 | 108.7 MiB | MNIST |
| Classification | Models | 16 | 96.7 MiB | resnet18-f37072fd |
| Tabular / Time Series | Datasets | 2 | 91.8 MiB | telecom |
| Segmentation | Models | 1 | 73.4 MiB | 2025_02_28 |
| Reranker | Models | 2 | 23.8 MiB | model |
| Experiment Log | Experiments & Logs | 30 | 16.4 MiB | archive |
| Detection Dataset | Datasets | 1 | 455.6 KiB | coco8 |

## Where the space has gone

### By drive

|  | Assets | Storage | Share |
| --- | ---: | ---: | ---: |
| C: | 92 | 27.8 GiB | 66.8% |
| F: | 62 | 13.8 GiB | 33.2% |

### By framework

|  | Assets | Storage | Share |
| --- | ---: | ---: | ---: |
| transformers | 16 | 30.2 GiB | 72.4% |
| ollama | 1 | 4.9 GiB | 11.7% |
| pytorch | 13 | 4.8 GiB | 11.6% |
| unknown | 95 | 1.4 GiB | 3.3% |
| onnxruntime | 16 | 305.4 MiB | 0.7% |
| peft | 2 | 125.8 MiB | 0.3% |
| paddle | 3 | 13.3 MiB | 0.0% |
| ultralytics | 6 | 10.6 MiB | 0.0% |
| tensorflow | 2 | 9.8 KiB | 0.0% |

### By storage format

|  | Assets | Storage | Share |
| --- | ---: | ---: | ---: |
| safetensors | 10 | 19.3 GiB | 46.4% |
| unknown | 45 | 8.8 GiB | 21.1% |
| pytorch | 20 | 8.1 GiB | 19.4% |
| gguf | 1 | 4.9 GiB | 11.7% |
| onnx | 16 | 305.4 MiB | 0.7% |
| tflite | 58 | 300.1 MiB | 0.7% |
| coreml | 1 | 15.5 MiB | 0.0% |
| paddle | 3 | 13.3 MiB | 0.0% |

### By task

|  | Assets | Storage | Share |
| --- | ---: | ---: | ---: |
| OCR | 11 | 15.5 GiB | 37.3% |
| Development | 4 | 8.4 GiB | 20.1% |
| Text Generation | 2 | 6.0 GiB | 14.4% |
| Checkpointing | 10 | 4.6 GiB | 11.1% |
| Visual Question Answering | 1 | 2.3 GiB | 5.5% |
| Embeddings | 4 | 1.4 GiB | 3.4% |
| Chat | 2 | 953.3 MiB | 2.2% |
| Speech Recognition | 6 | 817.2 MiB | 1.9% |
| Sentiment Analysis | 1 | 479.1 MiB | 1.1% |
| unclassified | 51 | 377.4 MiB | 0.9% |
| Image Classification | 18 | 205.3 MiB | 0.5% |
| Language Modelling | 4 | 195.5 MiB | 0.5% |
| Object Detection | 4 | 170.8 MiB | 0.4% |
| Fine-Tuning | 1 | 125.7 MiB | 0.3% |

## The twenty largest assets

| # | Name | Category | Size | Files | Location |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | thorn-nlp | AI Project | 8.4 GiB | 957 | `F:\project\NLP-Project\thorn-nlp` |
| 2 | baidu/Unlimited-OCR | OCR Model | 6.2 GiB | 11 | `C:\Users\pc\.cache\huggingface\hub\models--baidu--Unlimited-OCR` |
| 3 | deepseek-r1:8b | LLM | 4.9 GiB | 2 | `C:\Users\pc\.ollama\models\manifests\registry.ollama.ai\library\deepseek-r1\8b` |
| 4 | NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct | OCR Model | 4.1 GiB | 11 | `C:\Users\pc\.cache\huggingface\hub\models--NAMAA-Space--Qari-OCR-v0.3-VL-2B-Instruct` |
| 5 | zai-org/GLM-OCR | OCR Model | 2.5 GiB | 7 | `C:\Users\pc\.cache\huggingface\hub\models--zai-org--GLM-OCR` |
| 6 | unsloth/qwen2-vl-2b-instruct-unsloth-bnb-4bi | Vision-Language | 2.3 GiB | 3 | `C:\Users\pc\.cache\huggingface\hub\models--unsloth--qwen2-vl-2b-instruct-unsloth-bnb-4bit` |
| 7 | PaddlePaddle/PaddleOCR-VL | OCR Model | 1.8 GiB | 15 | `C:\Users\pc\.cache\huggingface\hub\models--PaddlePaddle--PaddleOCR-VL` |
| 8 | google/byt5-small | LLM | 1.1 GiB | 5 | `C:\Users\pc\.cache\huggingface\hub\models--google--byt5-small` |
| 9 | Qwen/Qwen2.5-0.5B-Instruct | LLM | 953.3 MiB | 7 | `C:\Users\pc\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct` |
| 10 | 2025_02_18 | OCR Model | 897.0 MiB | 10 | `C:\Users\pc\AppData\Local\datalab\datalab\Cache\models\text_recognition\2025_02_18` |
| 11 | large-v3-turbo | Speech | 660.0 MiB | 1 | `C:\Users\pc\.cache\whisper\large-v3-turbo.pt` |
| 12 | UBC-NLP/MARBERTv2 | Embedding | 624.9 MiB | 5 | `C:\Users\pc\.cache\huggingface\hub\models--UBC-NLP--MARBERTv2` |
| 13 | best_dapt | Checkpoint | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\best_dapt.pt` |
| 14 | dapt_epoch0 | Checkpoint | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch0.pt` |
| 15 | dapt_epoch1 | Checkpoint | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch1.pt` |
| 16 | dapt_epoch2 | Checkpoint | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch2.pt` |
| 17 | dapt_epoch3 | Checkpoint | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch3.pt` |
| 18 | dapt_epoch4 | Checkpoint | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch4.pt` |
| 19 | cardiffnlp/twitter-roberta-base-sentiment-la | Text Classification | 479.1 MiB | 5 | `C:\Users\pc\.cache\huggingface\hub\models--cardiffnlp--twitter-roberta-base-sentiment-latest` |
| 20 | nlpaueb/legal-bert-base-uncased | Embedding | 420.3 MiB | 5 | `C:\Users\pc\.cache\huggingface\hub\models--nlpaueb--legal-bert-base-uncased` |

## Models (101)

| Name | Category | Task | Family | Params | Quant | Format | Size | Health |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| baidu/Unlimited-OCR | OCR Model | OCR |  | 3.3B |  | safetensors | 6.2 GiB | 100 |
| deepseek-r1:8b | LLM | Text Generation | DeepSeek | 8.2B | Q4_K_M | gguf | 4.9 GiB | 100 |
| NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct | OCR Model | OCR | Qari | 2.2B |  | safetensors | 4.1 GiB | 100 |
| zai-org/GLM-OCR | OCR Model | OCR |  | 1.3B |  | safetensors | 2.5 GiB | 100 |
| unsloth/qwen2-vl-2b-instruct-unsloth-bnb | Vision-Language | Visual Question Answering | Qwen-VL | 1.6B | bnb-4bit-nf4 | safetensors | 2.3 GiB | 90 |
| PaddlePaddle/PaddleOCR-VL | OCR Model | OCR | PaddleOCR | 959M |  | safetensors | 1.8 GiB | 100 |
| google/byt5-small | LLM | Text Generation |  | ~300M |  | pytorch | 1.1 GiB | 100 |
| Qwen/Qwen2.5-0.5B-Instruct | LLM | Chat | Qwen | 494M |  | safetensors | 953.3 MiB | 100 |
| 2025_02_18 | OCR Model | OCR |  | 470M |  | safetensors | 897.0 MiB | 100 |
| large-v3-turbo | Speech | Speech Recognition | Whisper |  |  | pytorch | 660.0 MiB | 100 |
| UBC-NLP/MARBERTv2 | Embedding | Embeddings | MARBERT |  |  | pytorch | 624.9 MiB | 100 |
| cardiffnlp/twitter-roberta-base-sentimen | Text Classification | Sentiment Analysis | RoBERTa | ~125M |  | pytorch | 479.1 MiB | 100 |
| nlpaueb/legal-bert-base-uncased | Embedding | Embeddings | BERT |  |  | pytorch | 420.3 MiB | 100 |
| hf | Embedding | Embeddings | BERT | 110M |  | safetensors | 418.7 MiB | 100 |
| fasterrcnn_resnet50_fpn_coco-258fb6c6 | Object Detection | Object Detection | Faster R-CNN |  |  | pytorch | 159.7 MiB | 100 |
| rnnt_encode_chunks_quantized_dynamic | Speech | Speech Recognition |  |  |  | onnx | 143.3 MiB | 100 |
| NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instr | Adapter / LoRA | Fine-Tuning | qwen2-vl-2b-instruct-unsloth-bnb-4bit | 29M |  | safetensors | 125.7 MiB | 100 |
| model | Model |  |  |  |  | onnx | 73.5 MiB | 100 |
| 2025_02_28 | Segmentation | Semantic Segmentation |  | 38M |  | safetensors | 73.4 MiB | 100 |
| resnet18-f37072fd | Classification | Image Classification | ResNet | ~12M |  | pytorch | 44.7 MiB | 100 |
| model | Model |  |  |  |  | tflite | 35.1 MiB | 100 |
| model | Model |  |  |  |  | tflite | 35.1 MiB | 100 |
| model | Model |  |  |  |  | tflite | 35.1 MiB | 100 |
| model | Model |  |  |  |  | tflite | 35.1 MiB | 100 |
| arabic_historical | OCR Model | OCR | Kraken |  |  | coreml | 15.5 MiB | 100 |
| ivcam | Model |  |  |  |  | onnx | 14.3 MiB | 100 |
| word_fluency_v2 | Model |  |  |  |  | onnx | 13.5 MiB | 100 |
| model | Reranker | Reranking |  |  |  | onnx | 11.9 MiB | 100 |
| model | Reranker | Reranking |  |  |  | onnx | 11.9 MiB | 100 |
| mobilenet_v3_small-047dcff4 | Classification | Image Classification | MobileNet | ~3M |  | pytorch | 9.8 MiB | 100 |
| model | Model |  |  |  |  | onnx | 8.4 MiB | 100 |
| model | Model |  |  |  |  | onnx | 8.4 MiB | 100 |
| arabic_PP-OCRv4_rec_infer | OCR Model | OCR | PaddleOCR |  |  | paddle | 7.5 MiB | 100 |
| model | Model |  |  |  |  | tflite | 6.4 MiB | 100 |
| yolo11n | Object Detection | Object Detection | YOLO | ~3M |  | pytorch | 5.4 MiB | 100 |
| yolo26n | Object Detection | Object Detection | YOLO | ~3M |  | pytorch | 5.3 MiB | 100 |
| lid1_quantized_dynamic | Speech | Speech Recognition |  |  |  | onnx | 5.0 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| model | Model |  |  |  |  | tflite | 4.4 MiB | 100 |
| beng_deva_gujr_guru | Classification | Image Classification | ConvNeXt |  |  | tflite | 3.9 MiB | 100 |
| hanijpan | Classification | Image Classification | ConvNeXt |  |  | tflite | 3.7 MiB | 100 |
| Multilingual_PP-OCRv3_det_infer | OCR Model | OCR | PaddleOCR |  |  | paddle | 3.7 MiB | 100 |
| rnnt_joint_quantized_dynamic | Speech | Speech Recognition |  |  |  | onnx | 3.6 MiB | 100 |
| knda_sinh_telu | Classification | Image Classification | ConvNeXt |  |  | tflite | 3.6 MiB | 100 |
| khmr_laoo_thai | Classification | Image Classification | ConvNeXt |  |  | tflite | 3.5 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| visual_model_desktop | Model |  |  |  |  | tflite | 3.3 MiB | 100 |
| bede | Classification | Image Classification | ConvNeXt |  |  | tflite | 3.2 MiB | 100 |
| visual_model | Model |  |  |  |  | tflite | 3.2 MiB | 100 |
| kore | Classification | Image Classification | ConvNeXt |  |  | tflite | 3.0 MiB | 100 |
| rnnt_decoder_quantized_dynamic | Speech | Speech Recognition |  |  |  | onnx | 3.0 MiB | 100 |
| gujr | Classification | Image Classification | ConvNeXt |  |  | tflite | 2.9 MiB | 100 |
| gocr_mobile_und | OCR Model | OCR |  |  |  | tflite | 2.9 MiB | 100 |
| mlym | Classification | Image Classification | ConvNeXt |  |  | tflite | 2.7 MiB | 100 |
| arab | Classification | Image Classification | ConvNeXt |  |  | tflite | 2.7 MiB | 100 |
| cyrl | Classification | Image Classification | ConvNeXt |  |  | tflite | 2.6 MiB | 100 |
| hebr | Classification | Image Classification | ConvNeXt |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| model | Model |  |  |  |  | tflite | 2.6 MiB | 100 |
| grek | Classification | Image Classification | ConvNeXt |  |  | tflite | 2.6 MiB | 100 |
| taml | Classification | Image Classification | ConvNeXt |  |  | tflite | 2.6 MiB | 100 |
| geor | Classification | Image Classification | ConvNeXt |  |  | tflite | 2.6 MiB | 100 |
| rnnt_init_states | Speech | Speech Recognition |  |  |  | onnx | 2.3 MiB | 100 |
| ch_ppocr_mobile_v2.0_cls_infer | OCR Model | OCR | PaddleOCR |  |  | paddle | 2.1 MiB | 100 |
| WordCombinedFloatieLreOfflineExploration | Model |  |  |  |  | onnx | 2.1 MiB | 100 |
| gocr_group_rpn_text_detection_model_2024 | OCR Model | OCR |  |  |  | tflite | 2.0 MiB | 100 |
| PowerPointCombinedFloatieLreOnline | Model |  |  |  |  | onnx | 1.6 MiB | 100 |
| PowerPointFloatieTerminalV5 | Model |  |  |  |  | onnx | 1.4 MiB | 100 |
| WordCombinedFloatieLreOnlineV3 | Model |  |  |  |  | onnx | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| model | Model |  |  |  |  | tflite | 1.2 MiB | 100 |
| Qwen/Qwen1.5-4B-Chat | LLM | Chat | Qwen |  |  | unknown | 0 B | 75 |
| UBC-NLP/MARBERTv2 | Embedding | Embeddings | MARBERT |  |  | unknown | 0 B | 75 |

## Datasets (9)

| Name | Category | Format | Images | Classes | Splits | Size | Health |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| criminal | NLP Dataset | nlp |  |  |  | 190.4 MiB | 94 |
| MNIST | Image Dataset | image_classification |  |  |  | 63.5 MiB | 94 |
| telecom | Tabular / Time Series | tabular |  |  |  | 45.9 MiB | 94 |
| telecom | Tabular / Time Series | tabular |  |  |  | 45.9 MiB | 94 |
| hymenoptera_data | Image Dataset | image_classification | 398 | 2 | train, val | 45.2 MiB | 91 |
| knowledge_base | NLP Dataset | nlp |  |  |  | 3.3 MiB | 94 |
| c23fdff1a6bf74e0e1a71cb86f1e781d37da888c | NLP Dataset | hf_dataset |  |  |  | 1.7 MiB | 94 |
| coco8 | Detection Dataset | yolo | 8 | 1 | val, train | 455.6 KiB | 97 |
| wikimedia/wikipedia | NLP Dataset | hf_dataset |  |  | train | 127.8 KiB | 84 |

## Projects, runs and everything else (44)

| Name | Category | Framework | Size | Files | Location |
| --- | --- | --- | ---: | ---: | --- |
| thorn-nlp | AI Project | transformers | 8.4 GiB | 957 | `F:\project\NLP-Project\thorn-nlp` |
| best_dapt | Checkpoint | pytorch | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\best_dapt.pt` |
| dapt_epoch0 | Checkpoint | pytorch | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch0.pt` |
| dapt_epoch1 | Checkpoint | pytorch | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch1.pt` |
| dapt_epoch2 | Checkpoint | pytorch | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch2.pt` |
| dapt_epoch3 | Checkpoint | pytorch | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch3.pt` |
| dapt_epoch4 | Checkpoint | pytorch | 507.4 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch4.pt` |
| best_sc_dla | Checkpoint | pytorch | 419.7 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\sc_dla_old\best_sc_dla.pt` |
| best_sc_dla | Checkpoint | pytorch | 418.7 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\sc_dla\best_sc_dla.pt` |
| best_sc_dla | Checkpoint | pytorch | 418.2 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\sc_dla_pre_dapt\best_sc_dla.pt` |
| best_model | Checkpoint | pytorch | 417.7 MiB | 1 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\teacher\best_model.pt` |
| archive | Experiment Log | unknown | 6.7 MiB | 4 | `F:\project\MIA-Tasks\Phase 1\Task2\archive` |
| clause_detector | AI Project | transformers | 4.6 MiB | 75 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector` |
| run-20260515_201101-cwe4rwjy | Experiment Log | unknown | 3.3 MiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_201101-cwe4rwjy` |
| run-20260515_144644-zrswib49 | Experiment Log | unknown | 762.5 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_144644-zrswib49` |
| run-20260515_165711-aio37amm | Experiment Log | unknown | 735.2 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_165711-aio37amm` |
| run-20260515_125117-5k4mvjs9 | Experiment Log | unknown | 680.6 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_125117-5k4mvjs9` |
| run-20260515_013506-j1u3z1ja | Experiment Log | unknown | 513.5 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_013506-j1u3z1ja` |
| run-20260515_031643-urpytyn5 | Experiment Log | unknown | 498.1 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_031643-urpytyn5` |
| run-20260515_043840-ejrc2w32 | Experiment Log | unknown | 462.4 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_043840-ejrc2w32` |
| run-20260515_190251-131dej9z | Experiment Log | unknown | 427.1 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_190251-131dej9z` |
| run-20260515_200744-a1jrrffx | Experiment Log | unknown | 349.6 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb\run-20260515_200744-a1jrrffx` |
| run-20260516_111808-nbjgdrcc | Experiment Log | unknown | 304.6 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb\run-20260516_111808-nbjgdrcc` |
| run-20260516_140759-t0gid3zi | Experiment Log | unknown | 273.6 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260516_140759-t0gid3zi` |
| run-20260515_113419-ytyqn6x9 | Experiment Log | unknown | 251.1 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_113419-ytyqn6x9` |
| run-20260515_121316-w87i9hvy | Experiment Log | unknown | 246.1 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_121316-w87i9hvy` |
| run-20260515_110008-p0rbh4eq | Experiment Log | unknown | 224.8 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_110008-p0rbh4eq` |
| run-20260515_011641-dlokibah | Experiment Log | unknown | 208.6 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb\run-20260515_011641-dlokibah` |
| run-20260515_103122-bi6pcycf | Experiment Log | unknown | 140.5 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_103122-bi6pcycf` |
| thorn-nlp | AI Project | transformers | 133.6 KiB | 32 | `C:\Users\pc\My Drive\NLP-Project\thorn-nlp` |
| run-20260515_104548-ing5y249 | Experiment Log | unknown | 127.1 KiB | 8 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_104548-ing5y249` |
| run-20260515_101306-wpgxqbnj | Experiment Log | unknown | 114.4 KiB | 7 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_101306-wpgxqbnj` |
| libregrtest | AI Project | peft | 109.0 KiB | 11 | `C:\Users\pc\AppData\Local\Programs\Python\Python39\Lib\test\libregrtest` |
| run-20260515_101551-vyt7s65v | Experiment Log | unknown | 102.2 KiB | 7 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_101551-vyt7s65v` |
| run-20260515_055412-yvjcj6xn | Experiment Log | unknown | 52.6 KiB | 7 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_055412-yvjcj6xn` |
| run-20260516_140144-hvysa6wh | Experiment Log | unknown | 50.2 KiB | 7 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260516_140144-hvysa6wh` |
| run-20260516_105133-9blo6a6z | Experiment Log | unknown | 40.2 KiB | 4 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260516_105133-9blo6a6z` |
| run-20260516_140631-gp37atf9 | Experiment Log | unknown | 22.3 KiB | 7 | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260516_140631-gp37atf9` |
| test | Experiment Log | tensorflow | 6.3 KiB | 1 | `C:\Users\pc\OneDrive - Al Alamein University\GCI\04. Competition (due Jun 12th 11AM UTC)\competition\catboost_info\test` |
| learn | Experiment Log | tensorflow | 3.4 KiB | 1 | `C:\Users\pc\OneDrive - Al Alamein University\GCI\04. Competition (due Jun 12th 11AM UTC)\competition\catboost_info\learn` |
| train-2 | Experiment Log | ultralytics | 1.7 KiB | 1 | `F:\Downloads\runs\detect\train-2` |
| train-3 | Experiment Log | ultralytics | 1.7 KiB | 1 | `F:\Downloads\runs\detect\train-3` |
| train-4 | Experiment Log | ultralytics | 1.7 KiB | 1 | `F:\Downloads\runs\detect\train-4` |
| train | Experiment Log | ultralytics | 1.7 KiB | 1 | `F:\Downloads\runs\detect\train` |

## Relationships

| Relation | Edges | Meaning |
| --- | ---: | --- |
| belongs_to | 44 | asset lives inside a project or labelling workspace |
| derived_from | 1 | quantised or converted from another model |

### Every edge

| Asset | Relation | Target |
| --- | --- | --- |
| best_dapt | belongs_to | clause_detector |
| dapt_epoch0 | belongs_to | clause_detector |
| dapt_epoch1 | belongs_to | clause_detector |
| dapt_epoch2 | belongs_to | clause_detector |
| dapt_epoch3 | belongs_to | clause_detector |
| dapt_epoch4 | belongs_to | clause_detector |
| hf | belongs_to | clause_detector |
| best_sc_dla | belongs_to | clause_detector |
| best_sc_dla | belongs_to | clause_detector |
| best_sc_dla | belongs_to | clause_detector |
| best_model | belongs_to | clause_detector |
| c23fdff1a6bf74e0e1a71cb86f1e781d37da888c | belongs_to | clause_detector |
| run-20260515_013506-j1u3z1ja | belongs_to | clause_detector |
| run-20260515_031643-urpytyn5 | belongs_to | clause_detector |
| run-20260515_043840-ejrc2w32 | belongs_to | clause_detector |
| run-20260515_055412-yvjcj6xn | belongs_to | clause_detector |
| run-20260515_101306-wpgxqbnj | belongs_to | clause_detector |
| run-20260516_140144-hvysa6wh | belongs_to | clause_detector |
| run-20260516_140631-gp37atf9 | belongs_to | clause_detector |
| run-20260516_140759-t0gid3zi | belongs_to | clause_detector |
| run-20260515_011641-dlokibah | belongs_to | clause_detector |
| run-20260515_200744-a1jrrffx | belongs_to | clause_detector |
| run-20260516_111808-nbjgdrcc | belongs_to | clause_detector |
| run-20260515_101551-vyt7s65v | belongs_to | clause_detector |
| run-20260515_103122-bi6pcycf | belongs_to | clause_detector |
| run-20260515_104548-ing5y249 | belongs_to | clause_detector |
| run-20260515_110008-p0rbh4eq | belongs_to | clause_detector |
| run-20260515_113419-ytyqn6x9 | belongs_to | clause_detector |
| run-20260515_121316-w87i9hvy | belongs_to | clause_detector |
| run-20260515_125117-5k4mvjs9 | belongs_to | clause_detector |
| run-20260515_144644-zrswib49 | belongs_to | clause_detector |
| run-20260515_165711-aio37amm | belongs_to | clause_detector |
| run-20260515_190251-131dej9z | belongs_to | clause_detector |
| run-20260515_201101-cwe4rwjy | belongs_to | clause_detector |
| run-20260516_105133-9blo6a6z | belongs_to | clause_detector |
| knowledge_base | belongs_to | thorn-nlp |
| Qwen1.5-4B-Chat | belongs_to | thorn-nlp |
| MARBERTv2 | belongs_to | thorn-nlp |
| arabic_historical | belongs_to | thorn-nlp |
| ch_ppocr_mobile_v2.0_cls_infer | belongs_to | thorn-nlp |
| Multilingual_PP-OCRv3_det_infer | belongs_to | thorn-nlp |
| arabic_PP-OCRv4_rec_infer | belongs_to | thorn-nlp |
| criminal | belongs_to | thorn-nlp |
| clause_detector | belongs_to | thorn-nlp |
| Qari-OCR-0.2.2.1-VL-2B-Instruct | derived_from | qwen2-vl-2b-instruct-unsloth-bnb-4bit |

## Health

| Status | Assets |
| --- | ---: |
| error | 2 |
| ok | 150 |
| warning | 2 |

### What needs attention (4)

| Asset | Category | Score | Findings |
| --- | --- | ---: | --- |
| Qwen/Qwen1.5-4B-Chat | LLM | 75 | error: All 9 file(s) are zero bytes |
| UBC-NLP/MARBERTv2 | Embedding | 75 | error: All 5 file(s) are zero bytes |
| wikimedia/wikipedia | NLP Dataset | 84 | warning: No validation split<br>info: No licence file<br>info: No test split |
| unsloth/qwen2-vl-2b-instruct-unsloth | Vision-Language | 90 | warning: No tokenizer files |

## Complete listing

Every asset, alphabetically, with the path you can act on.

| Name | Category | Task | Size | Files | Drive | Path |
| --- | --- | --- | ---: | ---: | --- | --- |
| 2025_02_18 | OCR Model | OCR | 897.0 MiB | 10 | C: | `C:\Users\pc\AppData\Local\datalab\datalab\Cache\models\text_recognition\2025_02_18` |
| 2025_02_28 | Segmentation | Semantic Segmentation | 73.4 MiB | 7 | C: | `C:\Users\pc\AppData\Local\datalab\datalab\Cache\models\text_detection\2025_02_28` |
| arab | Classification | Image Classification | 2.7 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\arab.tflite` |
| arabic_historical | OCR Model | OCR | 15.5 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\data\kraken_cache\arabic_historical.mlmodel` |
| arabic_PP-OCRv4_rec_infer | OCR Model | OCR | 7.5 MiB | 3 | F: | `F:\project\NLP-Project\thorn-nlp\data\paddle_cache\whl\rec\arabic\arabic_PP-OCRv4_rec_infer` |
| archive | Experiment Log | Experiment Tracking | 6.7 MiB | 4 | F: | `F:\project\MIA-Tasks\Phase 1\Task2\archive` |
| baidu/Unlimited-OCR | OCR Model | OCR | 6.2 GiB | 11 | C: | `C:\Users\pc\.cache\huggingface\hub\models--baidu--Unlimited-OCR` |
| bede | Classification | Image Classification | 3.2 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\bede.tflite` |
| beng_deva_gujr_guru | Classification | Image Classification | 3.9 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\beng_deva_gujr_guru.tflite` |
| best_dapt | Checkpoint | Checkpointing | 507.4 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\best_dapt.pt` |
| best_model | Checkpoint | Checkpointing | 417.7 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\teacher\best_model.pt` |
| best_sc_dla | Checkpoint | Checkpointing | 419.7 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\sc_dla_old\best_sc_dla.pt` |
| best_sc_dla | Checkpoint | Checkpointing | 418.7 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\sc_dla\best_sc_dla.pt` |
| best_sc_dla | Checkpoint | Checkpointing | 418.2 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\sc_dla_pre_dapt\best_sc_dla.pt` |
| c23fdff1a6bf74e0e1a71cb86f1e781d37da888c | NLP Dataset | Language Modelling | 1.7 MiB | 4 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\data\cache\coastalcph___lex_glue\unfair_tos\0.0.0\c23fdff1a6bf74e0e1a71cb86f1e781d37da888c` |
| cardiffnlp/twitter-roberta-base-sentimen | Text Classification | Sentiment Analysis | 479.1 MiB | 5 | C: | `C:\Users\pc\.cache\huggingface\hub\models--cardiffnlp--twitter-roberta-base-sentiment-latest` |
| ch_ppocr_mobile_v2.0_cls_infer | OCR Model | OCR | 2.1 MiB | 3 | F: | `F:\project\NLP-Project\thorn-nlp\data\paddle_cache\whl\cls\ch_ppocr_mobile_v2.0_cls_infer` |
| clause_detector | AI Project | Development | 4.6 MiB | 75 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector` |
| coco8 | Detection Dataset | Object Detection | 455.6 KiB | 20 | F: | `F:\Downloads\datasets\coco8` |
| criminal | NLP Dataset | Language Modelling | 190.4 MiB | 5 | F: | `F:\project\NLP-Project\thorn-nlp\outputs\gemini_batch\criminal` |
| cyrl | Classification | Image Classification | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\cyrl.tflite` |
| dapt_epoch0 | Checkpoint | Checkpointing | 507.4 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch0.pt` |
| dapt_epoch1 | Checkpoint | Checkpointing | 507.4 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch1.pt` |
| dapt_epoch2 | Checkpoint | Checkpointing | 507.4 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch2.pt` |
| dapt_epoch3 | Checkpoint | Checkpointing | 507.4 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch3.pt` |
| dapt_epoch4 | Checkpoint | Checkpointing | 507.4 MiB | 1 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\dapt_epoch4.pt` |
| deepseek-r1:8b | LLM | Text Generation | 4.9 GiB | 2 | C: | `C:\Users\pc\.ollama\models\manifests\registry.ollama.ai\library\deepseek-r1\8b` |
| fasterrcnn_resnet50_fpn_coco-258fb6c6 | Object Detection | Object Detection | 159.7 MiB | 1 | C: | `C:\Users\pc\.cache\torch\hub\checkpoints\fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` |
| geor | Classification | Image Classification | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\geor.tflite` |
| gocr_group_rpn_text_detection_model_2024 | OCR Model | OCR | 2.0 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\detection\gocr_group_rpn_text_detection_model_2024_q4.tflite` |
| gocr_mobile_und | OCR Model | OCR | 2.9 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\gocr_mobile_und.tflite` |
| google/byt5-small | LLM | Text Generation | 1.1 GiB | 5 | C: | `C:\Users\pc\.cache\huggingface\hub\models--google--byt5-small` |
| grek | Classification | Image Classification | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\grek.tflite` |
| gujr | Classification | Image Classification | 2.9 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\gujr.tflite` |
| hanijpan | Classification | Image Classification | 3.7 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\hanijpan.tflite` |
| hebr | Classification | Image Classification | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\hebr.tflite` |
| hf | Embedding | Embeddings | 418.7 MiB | 7 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\artifacts\dapt_cuad\hf` |
| hymenoptera_data | Image Dataset | Image Classification | 45.2 MiB | 398 | F: | `F:\Downloads\hymenoptera_data` |
| ivcam | Model |  | 14.3 MiB | 1 | F: | `F:\New Apps\iVCam\resource\ivcam.onnx` |
| khmr_laoo_thai | Classification | Image Classification | 3.5 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\khmr_laoo_thai.tflite` |
| knda_sinh_telu | Classification | Image Classification | 3.6 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\knda_sinh_telu.tflite` |
| knowledge_base | NLP Dataset | Language Modelling | 3.3 MiB | 5 | F: | `F:\project\NLP-Project\thorn-nlp\data\civil_law\knowledge_base` |
| kore | Classification | Image Classification | 3.0 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\kore.tflite` |
| large-v3-turbo | Speech | Speech Recognition | 660.0 MiB | 1 | C: | `C:\Users\pc\.cache\whisper\large-v3-turbo.pt` |
| learn | Experiment Log | Experiment Tracking | 3.4 KiB | 1 | C: | `C:\Users\pc\OneDrive - Al Alamein University\GCI\04. Competition (due Jun 12th 11AM UTC)\competition\catboost_info\learn` |
| libregrtest | AI Project | Development | 109.0 KiB | 11 | C: | `C:\Users\pc\AppData\Local\Programs\Python\Python39\Lib\test\libregrtest` |
| lid1_quantized_dynamic | Speech | Speech Recognition | 5.0 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Zoom\asr\asr_model\embedded_asr_model_en\lid1_quantized_dynamic.onnx` |
| mlym | Classification | Image Classification | 2.7 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\mlym.tflite` |
| MNIST | Image Dataset | Image Classification | 63.5 MiB | 8 | F: | `F:\Downloads\data\MNIST\raw` |
| mobilenet_v3_small-047dcff4 | Classification | Image Classification | 9.8 MiB | 1 | C: | `C:\Users\pc\.cache\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth` |
| model | Model |  | 73.5 MiB | 1 | C: | `C:\Users\pc\.antigravity\extensions\devsense.intelli-php-vscode-0.12.17700-win32-x64\out\server\models\intelliphp_v3\model.onnx` |
| model | Model |  | 35.1 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\43\E6DC4029A1E4B4C1\E7C3ACF93CFA1282\model.tflite` |
| model | Model |  | 35.1 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\optimization_guide_model_store\43\E6DC4029A1E4B4C1\220F3D89A5096351\model.tflite` |
| model | Model |  | 35.1 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\369a5558a207e054bec6b3942b379b09\ms-vscode.js-debug\.profile\optimization_guide_model_store\43\E6DC4029A1E4B4C1\0A2121433BAB6940\model.tflite` |
| model | Model |  | 35.1 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\78c8b8dac8a7098800686f1c797c8cc0\ms-vscode.js-debug\.profile\optimization_guide_model_store\43\E6DC4029A1E4B4C1\BC0CAF68B218689E\model.tflite` |
| model | Reranker | Reranking | 11.9 MiB | 1 | C: | `C:\Users\pc\.antigravity\extensions\devsense.phptools-vscode-1.70.18840-win32-x64\out\server\models\deeprerank\model.onnx` |
| model | Reranker | Reranking | 11.9 MiB | 1 | C: | `C:\Users\pc\.antigravity\extensions\devsense.phptools-vscode-1.71.19014-win32-x64\out\server\models\deeprerank\model.onnx` |
| model | Model |  | 8.4 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Microsoft\AugLoop\Word\2.1\CloudResources\ModelResources\en-us\model.onnx` |
| model | Model |  | 8.4 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Microsoft\AugLoop\Word\CloudResources\ModelResources\textprediction_swiftkey\0.9\model.onnx` |
| model | Model |  | 6.4 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\optimization_guide_model_store\30\E6DC4029A1E4B4C1\EC3EEC558B6B3677\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\24\E6DC4029A1E4B4C1\181DA9D5A1513EAD\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\24\E6DC4029A1E4B4C1\393EE92F968E7467\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\optimization_guide_model_store\24\E6DC4029A1E4B4C1\FFA01AF4FE21B131\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\369a5558a207e054bec6b3942b379b09\ms-vscode.js-debug\.profile\optimization_guide_model_store\24\E6DC4029A1E4B4C1\3E361B7F48897184\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\78c8b8dac8a7098800686f1c797c8cc0\ms-vscode.js-debug\.profile\optimization_guide_model_store\24\E6DC4029A1E4B4C1\DEBB57122A105881\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\7cd6466ebe9189464826547a3416c58e\ms-vscode.js-debug\.profile\optimization_guide_model_store\24\E6DC4029A1E4B4C1\929CED4F815137AC\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\99aee63627c71dd5fe5f54b44cb02473\ms-vscode.js-debug\.profile\optimization_guide_model_store\24\E6DC4029A1E4B4C1\F5E23FB1715F5BB4\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\7cd6466ebe9189464826547a3416c58e\ms-vscode.js-debug\.profile\optimization_guide_model_store\24\E6DC4029A1E4B4C1\929CED4F815137AC\model.tflite` |
| model | Model |  | 4.4 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\99aee63627c71dd5fe5f54b44cb02473\ms-vscode.js-debug\.profile\optimization_guide_model_store\24\E6DC4029A1E4B4C1\F5E23FB1715F5BB4\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\15\E6DC4029A1E4B4C1\A7138637C5812324\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\15\E6DC4029A1E4B4C1\BCCF90EE72B0BD68\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\optimization_guide_model_store\15\E6DC4029A1E4B4C1\174A4748C01ACE5D\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\369a5558a207e054bec6b3942b379b09\ms-vscode.js-debug\.profile\optimization_guide_model_store\15\E6DC4029A1E4B4C1\7816052095F26134\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\78c8b8dac8a7098800686f1c797c8cc0\ms-vscode.js-debug\.profile\optimization_guide_model_store\15\E6DC4029A1E4B4C1\DDDBD65E07CE1F8E\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\7cd6466ebe9189464826547a3416c58e\ms-vscode.js-debug\.profile\optimization_guide_model_store\15\E6DC4029A1E4B4C1\E38B38E93F1E494E\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\99aee63627c71dd5fe5f54b44cb02473\ms-vscode.js-debug\.profile\optimization_guide_model_store\15\E6DC4029A1E4B4C1\48BC290EE0374C43\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\7cd6466ebe9189464826547a3416c58e\ms-vscode.js-debug\.profile\optimization_guide_model_store\15\E6DC4029A1E4B4C1\E38B38E93F1E494E\model.tflite` |
| model | Model |  | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\99aee63627c71dd5fe5f54b44cb02473\ms-vscode.js-debug\.profile\optimization_guide_model_store\15\E6DC4029A1E4B4C1\48BC290EE0374C43\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\13\E6DC4029A1E4B4C1\159507883C386940\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\13\E6DC4029A1E4B4C1\7A5D88AE278F1F53\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\optimization_guide_model_store\13\E6DC4029A1E4B4C1\AE3D4A74714EB2AE\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\369a5558a207e054bec6b3942b379b09\ms-vscode.js-debug\.profile\optimization_guide_model_store\13\E6DC4029A1E4B4C1\FA9BD0AF42462555\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\78c8b8dac8a7098800686f1c797c8cc0\ms-vscode.js-debug\.profile\optimization_guide_model_store\13\E6DC4029A1E4B4C1\BA450B883ABD3FD1\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\7cd6466ebe9189464826547a3416c58e\ms-vscode.js-debug\.profile\optimization_guide_model_store\13\E6DC4029A1E4B4C1\87924F911075A152\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\99aee63627c71dd5fe5f54b44cb02473\ms-vscode.js-debug\.profile\optimization_guide_model_store\13\E6DC4029A1E4B4C1\FC560612DFBFB862\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\7cd6466ebe9189464826547a3416c58e\ms-vscode.js-debug\.profile\optimization_guide_model_store\13\E6DC4029A1E4B4C1\87924F911075A152\model.tflite` |
| model | Model |  | 1.2 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\99aee63627c71dd5fe5f54b44cb02473\ms-vscode.js-debug\.profile\optimization_guide_model_store\13\E6DC4029A1E4B4C1\FC560612DFBFB862\model.tflite` |
| Multilingual_PP-OCRv3_det_infer | OCR Model | OCR | 3.7 MiB | 3 | F: | `F:\project\NLP-Project\thorn-nlp\data\paddle_cache\whl\det\ml\Multilingual_PP-OCRv3_det_infer` |
| NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instr | Adapter / LoRA | Fine-Tuning | 125.7 MiB | 10 | C: | `C:\Users\pc\.cache\huggingface\hub\models--NAMAA-Space--Qari-OCR-0.2.2.1-VL-2B-Instruct` |
| NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct | OCR Model | OCR | 4.1 GiB | 11 | C: | `C:\Users\pc\.cache\huggingface\hub\models--NAMAA-Space--Qari-OCR-v0.3-VL-2B-Instruct` |
| nlpaueb/legal-bert-base-uncased | Embedding | Embeddings | 420.3 MiB | 5 | C: | `C:\Users\pc\.cache\huggingface\hub\models--nlpaueb--legal-bert-base-uncased` |
| PaddlePaddle/PaddleOCR-VL | OCR Model | OCR | 1.8 GiB | 15 | C: | `C:\Users\pc\.cache\huggingface\hub\models--PaddlePaddle--PaddleOCR-VL` |
| PowerPointCombinedFloatieLreOnline | Model |  | 1.6 MiB | 1 | F: | `F:\tools\tools analisis\Office16\AI\PowerPointCombinedFloatieLreOnline.onnx` |
| PowerPointFloatieTerminalV5 | Model |  | 1.4 MiB | 1 | F: | `F:\tools\tools analisis\Office16\AI\PowerPointFloatieTerminalV5.onnx` |
| Qwen/Qwen1.5-4B-Chat | LLM | Chat | 0 B | 9 | F: | `F:\project\NLP-Project\thorn-nlp\data\hf_cache\hub\models--Qwen--Qwen1.5-4B-Chat` |
| Qwen/Qwen2.5-0.5B-Instruct | LLM | Chat | 953.3 MiB | 7 | C: | `C:\Users\pc\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct` |
| resnet18-f37072fd | Classification | Image Classification | 44.7 MiB | 1 | C: | `C:\Users\pc\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth` |
| rnnt_decoder_quantized_dynamic | Speech | Speech Recognition | 3.0 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Zoom\asr\asr_model\embedded_asr_model_en\rnnt_decoder_quantized_dynamic.onnx` |
| rnnt_encode_chunks_quantized_dynamic | Speech | Speech Recognition | 143.3 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Zoom\asr\asr_model\embedded_asr_model_en\rnnt_encode_chunks_quantized_dynamic.onnx` |
| rnnt_init_states | Speech | Speech Recognition | 2.3 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Zoom\asr\asr_model\embedded_asr_model_en\rnnt_init_states.onnx` |
| rnnt_joint_quantized_dynamic | Speech | Speech Recognition | 3.6 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Zoom\asr\asr_model\embedded_asr_model_en\rnnt_joint_quantized_dynamic.onnx` |
| run-20260515_011641-dlokibah | Experiment Log | Experiment Tracking | 208.6 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb\run-20260515_011641-dlokibah` |
| run-20260515_013506-j1u3z1ja | Experiment Log | Experiment Tracking | 513.5 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_013506-j1u3z1ja` |
| run-20260515_031643-urpytyn5 | Experiment Log | Experiment Tracking | 498.1 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_031643-urpytyn5` |
| run-20260515_043840-ejrc2w32 | Experiment Log | Experiment Tracking | 462.4 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_043840-ejrc2w32` |
| run-20260515_055412-yvjcj6xn | Experiment Log | Experiment Tracking | 52.6 KiB | 7 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_055412-yvjcj6xn` |
| run-20260515_101306-wpgxqbnj | Experiment Log | Experiment Tracking | 114.4 KiB | 7 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260515_101306-wpgxqbnj` |
| run-20260515_101551-vyt7s65v | Experiment Log | Experiment Tracking | 102.2 KiB | 7 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_101551-vyt7s65v` |
| run-20260515_103122-bi6pcycf | Experiment Log | Experiment Tracking | 140.5 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_103122-bi6pcycf` |
| run-20260515_104548-ing5y249 | Experiment Log | Experiment Tracking | 127.1 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_104548-ing5y249` |
| run-20260515_110008-p0rbh4eq | Experiment Log | Experiment Tracking | 224.8 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_110008-p0rbh4eq` |
| run-20260515_113419-ytyqn6x9 | Experiment Log | Experiment Tracking | 251.1 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_113419-ytyqn6x9` |
| run-20260515_121316-w87i9hvy | Experiment Log | Experiment Tracking | 246.1 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_121316-w87i9hvy` |
| run-20260515_125117-5k4mvjs9 | Experiment Log | Experiment Tracking | 680.6 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_125117-5k4mvjs9` |
| run-20260515_144644-zrswib49 | Experiment Log | Experiment Tracking | 762.5 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_144644-zrswib49` |
| run-20260515_165711-aio37amm | Experiment Log | Experiment Tracking | 735.2 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_165711-aio37amm` |
| run-20260515_190251-131dej9z | Experiment Log | Experiment Tracking | 427.1 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_190251-131dej9z` |
| run-20260515_200744-a1jrrffx | Experiment Log | Experiment Tracking | 349.6 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb\run-20260515_200744-a1jrrffx` |
| run-20260515_201101-cwe4rwjy | Experiment Log | Experiment Tracking | 3.3 MiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260515_201101-cwe4rwjy` |
| run-20260516_105133-9blo6a6z | Experiment Log | Experiment Tracking | 40.2 KiB | 4 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb_src\run-20260516_105133-9blo6a6z` |
| run-20260516_111808-nbjgdrcc | Experiment Log | Experiment Tracking | 304.6 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\wandb\run-20260516_111808-nbjgdrcc` |
| run-20260516_140144-hvysa6wh | Experiment Log | Experiment Tracking | 50.2 KiB | 7 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260516_140144-hvysa6wh` |
| run-20260516_140631-gp37atf9 | Experiment Log | Experiment Tracking | 22.3 KiB | 7 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260516_140631-gp37atf9` |
| run-20260516_140759-t0gid3zi | Experiment Log | Experiment Tracking | 273.6 KiB | 8 | F: | `F:\project\NLP-Project\thorn-nlp\services\clause_detector\notebooks\wandb\run-20260516_140759-t0gid3zi` |
| taml | Classification | Image Classification | 2.6 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\screen_ai\148.12\gocr\gocr_models\line_recognition_mobile_convnext320_omni\taml.tflite` |
| telecom | Tabular / Time Series | Tabular Classification | 45.9 MiB | 2 | C: | `C:\Users\pc\OneDrive - Al Alamein University\GCI\05. Final Assignment (due Jun 19th 11AM UTC)\Final Assignment\telecom` |
| telecom | Tabular / Time Series | Tabular Classification | 45.9 MiB | 2 | F: | `F:\project\05. Final Assignment (due Jun 19th 11AM UTC)\Final Assignment\telecom` |
| test | Experiment Log | Experiment Tracking | 6.3 KiB | 1 | C: | `C:\Users\pc\OneDrive - Al Alamein University\GCI\04. Competition (due Jun 12th 11AM UTC)\competition\catboost_info\test` |
| thorn-nlp | AI Project | Development | 8.4 GiB | 957 | F: | `F:\project\NLP-Project\thorn-nlp` |
| thorn-nlp | AI Project | Development | 133.6 KiB | 32 | C: | `C:\Users\pc\My Drive\NLP-Project\thorn-nlp` |
| train | Experiment Log | Experiment Tracking | 1.7 KiB | 1 | F: | `F:\Downloads\runs\detect\train` |
| train-2 | Experiment Log | Experiment Tracking | 1.7 KiB | 1 | F: | `F:\Downloads\runs\detect\train-2` |
| train-3 | Experiment Log | Experiment Tracking | 1.7 KiB | 1 | F: | `F:\Downloads\runs\detect\train-3` |
| train-4 | Experiment Log | Experiment Tracking | 1.7 KiB | 1 | F: | `F:\Downloads\runs\detect\train-4` |
| UBC-NLP/MARBERTv2 | Embedding | Embeddings | 624.9 MiB | 5 | C: | `C:\Users\pc\.cache\huggingface\hub\models--UBC-NLP--MARBERTv2` |
| UBC-NLP/MARBERTv2 | Embedding | Embeddings | 0 B | 5 | F: | `F:\project\NLP-Project\thorn-nlp\data\hf_cache\hub\models--UBC-NLP--MARBERTv2` |
| unsloth/qwen2-vl-2b-instruct-unsloth-bnb | Vision-Language | Visual Question Answering | 2.3 GiB | 3 | C: | `C:\Users\pc\.cache\huggingface\hub\models--unsloth--qwen2-vl-2b-instruct-unsloth-bnb-4bit` |
| visual_model | Model |  | 3.2 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\ClientSidePhishing\30.2\visual_model.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\25\E6DC4029A1E4B4C1\26BBF78F67BD99E4\visual_model_desktop.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\.gemini\antigravity-browser-profile\optimization_guide_model_store\25\E6DC4029A1E4B4C1\D01C3EC3D57A0038\visual_model_desktop.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\AppData\Local\Google\Chrome\User Data\optimization_guide_model_store\25\E6DC4029A1E4B4C1\5433DF4004BBC841\visual_model_desktop.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\369a5558a207e054bec6b3942b379b09\ms-vscode.js-debug\.profile\optimization_guide_model_store\25\E6DC4029A1E4B4C1\E78C315A35D1DEAF\visual_model_desktop.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\78c8b8dac8a7098800686f1c797c8cc0\ms-vscode.js-debug\.profile\optimization_guide_model_store\25\E6DC4029A1E4B4C1\00A614592D067497\visual_model_desktop.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\7cd6466ebe9189464826547a3416c58e\ms-vscode.js-debug\.profile\optimization_guide_model_store\25\E6DC4029A1E4B4C1\4E87802F89C21091\visual_model_desktop.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Code\User\workspaceStorage\99aee63627c71dd5fe5f54b44cb02473\ms-vscode.js-debug\.profile\optimization_guide_model_store\25\E6DC4029A1E4B4C1\5FCDA52788DF9941\visual_model_desktop.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\7cd6466ebe9189464826547a3416c58e\ms-vscode.js-debug\.profile\optimization_guide_model_store\25\E6DC4029A1E4B4C1\4E87802F89C21091\visual_model_desktop.tflite` |
| visual_model_desktop | Model |  | 3.3 MiB | 1 | C: | `C:\Users\pc\AppData\Roaming\Cursor\User\workspaceStorage\99aee63627c71dd5fe5f54b44cb02473\ms-vscode.js-debug\.profile\optimization_guide_model_store\25\E6DC4029A1E4B4C1\5FCDA52788DF9941\visual_model_desktop.tflite` |
| wikimedia/wikipedia | NLP Dataset | Language Modelling | 127.8 KiB | 1 | C: | `C:\Users\pc\.cache\huggingface\hub\datasets--wikimedia--wikipedia` |
| word_fluency_v2 | Model |  | 13.5 MiB | 1 | F: | `F:\tools\tools analisis\Office16\AI\word_fluency_v2.onnx` |
| WordCombinedFloatieLreOfflineExploration | Model |  | 2.1 MiB | 1 | F: | `F:\tools\tools analisis\Office16\AI\WordCombinedFloatieLreOfflineExplorationV3.onnx` |
| WordCombinedFloatieLreOnlineV3 | Model |  | 1.2 MiB | 1 | F: | `F:\tools\tools analisis\Office16\AI\WordCombinedFloatieLreOnlineV3.onnx` |
| yolo11n | Object Detection | Object Detection | 5.4 MiB | 1 | F: | `F:\Downloads\yolo11n.pt` |
| yolo26n | Object Detection | Object Detection | 5.3 MiB | 1 | F: | `F:\Downloads\yolo26n.pt` |
| zai-org/GLM-OCR | OCR Model | OCR | 2.5 GiB | 7 | C: | `C:\Users\pc\.cache\huggingface\hub\models--zai-org--GLM-OCR` |

## How this was produced

```console
$ aam scan --add C:\ F:\
$ aam inventory
```

Assets are identified by **structure**, not by folder name. A directory holding `annotations/instances_train.json` with `images`, `annotations` and `categories` keys is a COCO dataset whatever it is called; a folder called `coco` holding three screenshots is not.

No model file was loaded to produce this. Parameter counts come from parsing safetensors headers and GGUF key-value blocks as raw bytes, and `.pt` files are read through their ZIP central directory rather than unpickled, because unpickling a checkpoint executes arbitrary code.

