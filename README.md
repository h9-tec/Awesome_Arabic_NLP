
# Awesome Arabic NLP

A curated list of awesome Arabic Natural Language Processing (NLP) resources, including the latest research, models, datasets, and tools. This repository is designed to be a comprehensive and up-to-date reference for researchers and developers in the Arabic NLP field.

## Table of Contents

- [Key Organizations](#key-organizations)
- [Benchmarks & Leaderboards](#benchmarks--leaderboards)
- [State-of-the-Art Models](#state-of-the-art-models)
  - [Large Language Models (LLMs)](#large-language-models-llms)
  - [Multimodal Models](#multimodal-models)
  - [Transformer-based Models](#transformer-based-models)
  - [Embedding Models](#embedding-models)
  - [Task-Specific Models](#task-specific-models)
- [Audio Models](#audio-models)
- [Vision Models](#vision-models)
- [Diacritization (Tashkeel)](#diacritization-tashkeel)
- [Dialect Identification](#dialect-identification)
- [Key Datasets](#key-datasets)
- [Essential Tools & Libraries](#essential-tools--libraries)
- [Top Research Papers & Conferences](#top-research-papers--conferences)
- [Tutorials & Learning Resources](#tutorials--learning-resources)
- [Companies & Startups](#companies--startups)
- [Awesome Lists](#awesome-lists)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Key Organizations

| Organization | Focus | Key Contributions | Link |
|---|---|---|---|
| **AUB MIND Lab** | Foundational Arabic NLP models | AraBERT, AraGPT2, AraELECTRA | [GitHub](https://github.com/aub-mind) |
| **CAMeL Lab, NYUAD** | Arabic NLP tools and models | CAMeLBERT, camel_tools | [GitHub](https://github.com/CAMeL-Lab) |
| **UBC-NLP** | Dialectal Arabic, multimodal models | MARBERT, AraT5, NileChat, PEARL, Dallah | [Hugging Face](https://huggingface.co/UBC-NLP) |
| **QCRI** | Arabic LLMs, knowledge editing | Fanar LLMs, AraDiCE, Farasa | [Hugging Face](https://huggingface.co/QCRI) |
| **SILMA AI** | State-of-the-art Arabic LLMs | SILMA LLMs, Arabic Broad Benchmark | [Website](https://silma.ai/) |
| **MBZUAI** | Multimodal and speech models | AIN, ArTST, ClArTTS | [Hugging Face](https://huggingface.co/MBZUAI) |
| **ARBML** | Democratizing Arabic NLP | masader, klaam, tkseem | [GitHub](https://github.com/ARBML) |
| **NAMAA-Space** | OCR and Egyptian Arabic models | Qari-OCR, EgypTalk-ASR | [Hugging Face](https://huggingface.co/NAMAA-Space) |
| **Omartificial-Intelligence-Space** | Arabic embedding models | GATE, Matryoshka embeddings | [Hugging Face](https://huggingface.co/Omartificial-Intelligence-Space) |
| **TII (Technology Innovation Institute)** | Arabic LLM benchmarks, Falcon | Open Arabic LLM Leaderboard, Falcon LLM | [Website](https://www.tii.ae/) |
| **SDAIA (Saudi Data & AI Authority)** | Sovereign Arabic LLM | ALLaM model | [Website](https://sdaia.gov.sa/) |
| **SinaLab, Birzeit University** | Arabic NLP tools and datasets | SinaTools, Wojood NER | [GitHub](https://github.com/SinaLab) |
| **G42 / Inception AI** | Arabic-centric LLMs | Jais LLM family | [Website](https://www.g42.ai/) |
| **FreedomIntelligence** | Arabic LLMs and alignment | AceGPT, Arabic cultural datasets | [GitHub](https://github.com/FreedomIntelligence) |
| **Helsinki-NLP** | Machine translation models | OPUS-MT Arabic translation models | [Hugging Face](https://huggingface.co/Helsinki-NLP) |
| **LightOn AI** | Arabic web data | ArabicWeb24 corpus (39B+ tokens) | [Hugging Face](https://huggingface.co/lightonai) |

## Benchmarks & Leaderboards

| Benchmark | Description | Link |
|---|---|---|
| **MTEB Arabic Leaderboard** | Massive Text Embedding Benchmark for Arabic | [Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) |
| **Arabic Broad Leaderboard (ABL)** | NextGen Evaluation Benchmark and Leaderboard for Arabic LLMs by SILMA AI | [Hugging Face](https://huggingface.co/spaces/silma-ai/Arabic-LLM-Broad-Leaderboard) |
| **Open Arabic LLM Leaderboard** | Evaluation of Arabic LLMs on a set of benchmarks | [Hugging Face](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard) |
| **ALUE** | Arabic Language Understanding Evaluation benchmark | [Website](https://www.alue.org/) |
| **BALSAM** | Benchmark of Arabic Language AI Systems and Models | [Website](https://benchmarks.ksaa.gov.sa/) |
| **SILMA RAGQA Benchmark** | Evaluates Arabic/English LMs in Extractive QA tasks | [Hugging Face](https://huggingface.co/datasets/silma-ai/silma-rag-qa-benchmark-v1.0) |
| **Arabic Broad Benchmark (ABB)** | Comprehensive dataset and evaluation tool for Arabic LLMs by SILMA AI | [Hugging Face](https://huggingface.co/datasets/silma-ai/arabic-broad-benchmark) |
| **ArabicMMLU** | Multi-task language understanding benchmark from school exams | [Hugging Face](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) |
| **GATmath and GATLc** | Benchmarks from Saudi GAT exams | [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0329129) |
| **ArabicRAGB** | Arabic Retrieval-Augmented Generation Benchmark (multi-dialect) | [Hugging Face](https://huggingface.co/datasets/HeshamHaroon/ArabicRAGB) |
| **ACVA** | Arabic Cultural Value Alignment benchmark (8000+ questions across 58 areas) | [Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment) |

## State-of-the-Art Models

### Large Language Models (LLMs)

| Model | Parameters | Developed By | Key Features | Link |
|---|---|---|---|---|
| **Jais** | 13B, 30B | Inception AI, Cerebras | Arabic-centric, bilingual (Arabic/English), instruction-tuned chat version | [Hugging Face](https://huggingface.co/inceptionai/jais-30b-v3) |
| **SILMA 1.0** | 9B | SILMA AI | Top-ranked Arabic LLM built on Google Gemma | [Hugging Face](https://huggingface.co/silma-ai/SILMA-9B-Instruct-v1.0) |
| **ALLaM** | 7B | SDAIA & IBM | Saudi's Sovereign Model, enterprise-focused | [Hugging Face](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview) |
| **Fanar-1-9B** | 9B | QCRI | Arabic-English LLM | [Hugging Face](https://huggingface.co/QCRI/Fanar-1-9B-Instruct) |
| **AceGPT** | 7B | FreedomIntelligence | Top performance on Arabic benchmarks, culturally aligned | [GitHub](https://github.com/FreedomIntelligence/AceGPT) |
| **Atlas-Chat** | 2B, 9B, 27B | MBZUAI-IFM Paris Lab | LLM family for Moroccan (Darija) dialect | [Hugging Face](https://huggingface.co/collections/MBZUAI-Paris/atlas-chat) |
| **NileChat-3B** | 3B | UBC-NLP | LLM for Egyptian and Moroccan dialects | [Hugging Face](https://huggingface.co/UBC-NLP/NileChat-3B) |
| **Nile-Chat** | 4B, 3x4B-A6B, 12B | MBZUAI-IFM Paris Lab | LLM family for Egyptian Arabic and Latin (Arabizi) scripts | [Hugging Face](https://huggingface.co/collections/MBZUAI-Paris/nile-chat) |
| **AraGPT2** | 1.5B | AUB MIND Lab | GPT-2 for Arabic text generation | [Hugging Face](https://huggingface.co/aubmindlab/aragpt2-mega) |
| **Command R7B Arabic** | 7B | Cohere | Arabic-optimized variant of Command R | [Hugging Face](https://huggingface.co/CohereLabs/c4ai-command-r7b-arabic-02-2025) |
| **SambaLingo-Arabic** | 7B, 70B | SambaNova Systems | Arabic-adapted Llama 2 with continued pretraining | [Hugging Face](https://huggingface.co/sambanovasystems/SambaLingo-Arabic-Chat) |
| **ArabianGPT** | - | - | Native Arabic GPT-based LLM | [Paper](https://arxiv.org/abs/2402.15313) |
| **Llama 3.3** | 70B | Meta | Strong Arabic performance | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| **Qwen 3** | 0.6B-235B | Alibaba | Multilingual model with Arabic support | [Hugging Face](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) |
| **Gemma 3** | 1B-27B | Google | Multimodal capabilities | [Hugging Face](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) |
| **Cohere command-a-03-2025** | 111B | Cohere | Optimized for RAG | [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-a-03-2025) |
| **Mistral Saba** | 24B | Mistral | Commercial API | [Website](https://mistral.ai/news/mistral-saba) |

### Multimodal Models

| Model | Parameters | Developed By | Key Features | Link |
|---|---|---|---|---|
| **AIN** | 8B | MBZUAI | Arabic-centric, inclusive Large Multimodal Model (LMM) | [GitHub](https://github.com/mbzuai-oryx/AIN) |
| **Dallah** | - | UBC-NLP | Advanced multimodal LLM for Arabic | [Hugging Face](https://huggingface.co/UBC-NLP/dallah) |

### Transformer-based Models

| Model | Developed By | Key Features | Link |
|---|---|---|---|
| **AraBERT** | AUB MIND Lab | First BERT model for Arabic, multiple versions and sizes | [GitHub](https://github.com/aub-mind/arabert) |
| **AraBERTv02** | AUB MIND Lab | Updated AraBERT with improved tokenization (135M params, 12.4M+ downloads) | [Hugging Face](https://huggingface.co/aubmindlab/bert-base-arabertv02) |
| **CAMeLBERT** | CAMeL Lab, NYUAD | Models for MSA, Dialectal, and Classical Arabic | [GitHub](https://github.com/CAMeL-Lab/CAMeLBERT) |
| **MARBERT** | UBC-NLP | Focused on Dialectal Arabic and MSA | [GitHub](https://github.com/UBC-NLP/marbert) |
| **MARBERTv2** | UBC-NLP | Updated MARBERT with improved dialectal coverage | [Hugging Face](https://huggingface.co/UBC-NLP/MARBERTv2) |
| **AraELECTRA** | AUB MIND Lab | ELECTRA for Arabic | [Hugging Face](https://huggingface.co/aubmindlab/araelectra-base-discriminator) |
| **AraT5** | UBC-NLP | T5 model for Arabic (summarization, translation, paraphrasing) | [Hugging Face](https://huggingface.co/UBC-NLP/AraT5-base) |

### Embedding Models

| Model | Developer | Key Features | Link |
|---|---|---|---|
| **GATE-AraBert-v1** | Omartificial-Intelligence-Space | General Arabic Text Embedding, SOTA on MTEB Arabic STS | [Hugging Face](https://huggingface.co/Omartificial-Intelligence-Space/GATE-AraBert-v1) |
| **Arabic-Triplet-Matryoshka-V2** | Omartificial-Intelligence-Space | Matryoshka representation for efficient embedding | [Hugging Face](https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2) |
| **Swan** | UBC-NLP | Dialect-aware and cross-lingual (Small & Large versions) | [Paper](https://arxiv.org/abs/2411.01192) |
| **asafaya/bert-base-arabic** | asafaya | BERT-based Arabic embeddings | [Hugging Face](https://huggingface.co/asafaya/bert-base-arabic) |
| **ModernBERT-Arabic-Embeddings** | BounharAbdelaziz | ModernBERT-based Arabic sentence embeddings | [Hugging Face](https://huggingface.co/BounharAbdelaziz/ModernBERT-Arabic-Embeddings) |
| **DIMI-embedding-matryoshka-arabic** | AhmedZaky1 | Matryoshka embeddings based on AraBERT for NLI tasks | [Hugging Face](https://huggingface.co/AhmedZaky1/DIMI-embedding-matryoshka-arabic) |

### Task-Specific Models

| Model | Task | Key Features | Link |
|---|---|---|---|
| **CAMeLBERT-MSA-Sentiment** | Sentiment Analysis | Fine-tuned CAMeLBERT for MSA sentiment classification | [Hugging Face](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment) |
| **t5-arabic-text-summarization** | Summarization | T5 fine-tuned for Arabic news summarization | [Hugging Face](https://huggingface.co/malmarjeh/t5-arabic-text-summarization) |
| **opus-mt-en-ar** | Translation (EN→AR) | Helsinki-NLP OPUS machine translation (3.5M+ downloads) | [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) |
| **opus-mt-ar-en** | Translation (AR→EN) | Helsinki-NLP OPUS machine translation (12.4M+ downloads) | [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en) |
| **arabic-gec-v1** | Grammar Correction | Gemma-3-1b fine-tuned for Arabic grammatical error correction | [Hugging Face](https://huggingface.co/alnnahwi/gemma-3-1b-arabic-gec-v1) |
| **Arabic-Text-Correction** | Text Correction | AraT5-based Arabic text correction | [Hugging Face](https://huggingface.co/SuperSl6/Arabic-Text-Correction) |
| **arat5-arabic-dialects-translation** | Dialect Translation | AraT5 for translating Arabic dialects to MSA | [Hugging Face](https://huggingface.co/PRAli22/arat5-arabic-dialects-translation) |

## Audio Models

### Automatic Speech Recognition (ASR)

| Model | Key Features | Link |
|---|---|---|
| **openai/whisper-large-v3** | Supports Arabic among many languages | [Hugging Face](https://huggingface.co/openai/whisper-large-v3) |
| **MasriSwitch-Gemma3n-Transcriber-v1** | Egyptian Arabic code-switching transcription | [Hugging Face](https://huggingface.co/oddadmix/MasriSwitch-Gemma3n-Transcriber-v1) |
| **wav2vec2-large-xlsr-53-arabic** | Fine-tuned on Common Voice 6.1 and Arabic Speech Corpus | [Hugging Face](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic) |
| **artst_asr_v3** | ArTST model for ASR on MGB2 (best for MSA) | [Hugging Face](https://huggingface.co/MBZUAI/artst_asr_v3) |
| **EgypTalk-ASR-v2** | High-performance ASR for Egyptian Arabic | [Hugging Face](https://huggingface.co/NAMAA-Space/EgypTalk-ASR-v2) |

### Text-to-Speech (TTS)

| Model | Key Features | Link |
|---|---|---|
| **facebook/mms-tts-ara** | Arabic TTS from Facebook's Massively Multilingual Speech | [Hugging Face](https://huggingface.co/facebook/mms-tts-ara) |
| **speecht5_tts_clartts_ar** | SpeechT5 for Classical Arabic TTS | [Hugging Face](https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar) |
| **F5-TTS-Arabic** | F5-TTS fine-tuned for Arabic with regional diversity | [Hugging Face](https://huggingface.co/IbrahimSalah/F5-TTS-Arabic) |

## Vision Models

### Optical Character Recognition (OCR)

| Model | Key Features | Link |
|---|---|---|
| **Qari-OCR-0.1-VL-2B-Instruct** | Built on Qwen2 VL 2B, fine-tuned for Arabic OCR | [Hugging Face](https://huggingface.co/NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct) |
| **arabic-large-nougat** | End-to-end structured OCR for Arabic | [Hugging Face](https://huggingface.co/MohamedRashad/arabic-large-nougat) |

### Image Captioning

| Model | Key Features | Link |
|---|---|---|
| **blip-Arabic-flickr-8k** | BLIP fine-tuned for Arabic image captioning | [Hugging Face](https://huggingface.co/omarsabri8756/blip-Arabic-flickr-8k) |

## Diacritization (Tashkeel)

Models and tools for Arabic text diacritization (adding harakat/tashkeel to Arabic text).

### Models

| Model / System | Key Features | Link |
|---|---|---|
| **CATT** | Character-based Arabic Tashkeel Transformer, SOTA on WikiNews and CATT datasets | [Paper](https://arxiv.org/abs/2407.03236) |
| **Fine-Tashkeel** | Fine-tuned ByT5 for Arabic diacritization, 40% WER reduction | [Paper](https://www.researchgate.net/publication/372616004) |
| **Sadeed** | Small language model approach to Arabic diacritization | [Paper](https://arxiv.org/abs/2504.21635) |
| **Shakkala** | Neural Arabic text vocalization using bidirectional LSTM | [GitHub](https://github.com/AliOsm/shakkelha) |
| **Mishkal** | Rule-based diacritizer using affix detection and dictionary lookups | [GitHub](https://github.com/linuxscout/mishkal) |

### Datasets

| Dataset | Description | Link |
|---|---|---|
| **Tashkeela** | Arabic diacritization corpus from Kaggle | [GitHub](https://github.com/Anwarvic/Arabic-Tashkeela-Model) |
| **arabic-text-diacritization** | Benchmark dataset with helpers and systems comparison | [GitHub](https://github.com/AliOsm/arabic-text-diacritization) |

## Dialect Identification

Resources for identifying and classifying Arabic dialects.

### Shared Tasks

| Task | Description | Link |
|---|---|---|
| **NADI 2025** | First Multidialectal Arabic Speech Processing Shared Task (8-way dialect classification + ASR) | [Website](https://nadi.dlnlp.ai/2025/) |
| **NADI 2024** | Fifth Nuanced Arabic Dialect Identification Shared Task (multi-label DID, dialectness, dialect-to-MSA translation) | [Paper](https://arxiv.org/abs/2407.04910) |
| **NADI Shared Tasks** | Ongoing series of Arabic Dialect Identification shared tasks | [Website](https://nadi.dlnlp.ai/) |

### Datasets

| Dataset | Description | Link |
|---|---|---|
| **QADI (Arabic Dialect Identification)** | Automatically collected tweets dataset for multi-class dialect classification | [Hugging Face](https://huggingface.co/datasets/Abdelrahman-Rezk/Arabic_Dialect_Identification) |
| **Arabic POS Dialect** | Part-of-speech tagging in Arabic dialects (350 manually segmented sentences per dialect) | [Hugging Face](https://huggingface.co/datasets/QCRI/arabic_pos_dialect) |
| **Arabic Dialects to MSA** | Parallel corpus of Arabic dialects and their MSA translations | [Hugging Face](https://huggingface.co/datasets/PRAli22/Arabic_dialects_to_MSA) |
| **Casablanca** | Comprehensive multidialectal Arabic speech dataset used in NADI 2025 | [Paper](https://arxiv.org/abs/2410.04527) |

## Key Datasets

### Text Datasets

| Dataset | Description | Link |
|---|---|---|
| **masader** | Largest public catalogue of Arabic NLP/speech datasets (600+) | [GitHub](https://github.com/ARBML/masader) |
| **Wojood** | Nested Arabic Named Entity Recognition (NER) corpus (550K tokens) | [GitHub](https://github.com/SinaLab/ArabicNER) |
| **CIDAR** | Culturally relevant instruction dataset (10,000 pairs) | [Hugging Face](https://huggingface.co/datasets/arbml/CIDAR) |
| **Gazelle** | Arabic writing assistance dataset | [Paper](https://huggingface.co/papers/2410.18163) |
| **The Arabic E-Book Corpus** | 1,745 books (81.5M words) from the Hindawi foundation | [Hugging Face](https://huggingface.co/datasets/mohres/The_Arabic_E-Book_Corpus) |
| **ArabicCorpus2B** | 1.9B word Arabic corpus from various sources | [Hugging Face](https://huggingface.co/datasets/tarekeldeeb/ArabicCorpus2B) |
| **BAREC Corpus** | Arabic Readability Assessment Corpus | [Hugging Face](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0) |
| **palm** | Comprehensive human-created Arabic instruction dataset | [Hugging Face](https://huggingface.co/datasets/UBC-NLP/palm) |
| **ArabicaQA** | Large-scale dataset for Arabic Question Answering | [GitHub](https://github.com/DataScienceUIBK/ArabicaQA) |
| **arabic-hate-speech-superset** | Comprehensive hate speech detection dataset | [Hugging Face](https://huggingface.co/datasets/manueltonneau/arabic-hate-speech-superset) |
| **dialogue-arabic-dialects** | Message-response pairs in Levantine, Egyptian, Gulf dialects | [GitHub](https://github.com/tareknaous/dialogue-arabic-dialects) |
| **101 Billion Arabic Words** | Massive Arabic web corpus (101B words) | [Hugging Face](https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset) |
| **ArabicText-Large** | High-quality Arabic corpus (743K articles) for LLM training | [Hugging Face](https://huggingface.co/datasets/Jr23xd23/ArabicText-Large) |
| **ArabicWeb24** | 39B+ tokens of high-quality Arabic web content | [Hugging Face](https://huggingface.co/datasets/lightonai/ArabicWeb24) |
| **Arabic_Function_Calling** | First Arabic function calling dataset (50K+ samples, multi-dialect) | [Hugging Face](https://huggingface.co/datasets/HeshamHaroon/Arabic_Function_Calling) |
| **Mixed Arabic Datasets (MAD)** | Community-driven collection of diverse Arabic texts | [Hugging Face](https://huggingface.co/datasets/M-A-D/Mixed-Arabic-Datasets-Repo) |
| **Arabic-OpenHermes-2.5** | Arabic translation of OpenHermes instruction dataset | [Hugging Face](https://huggingface.co/datasets/2A2I/Arabic-OpenHermes-2.5) |
| **Alpaca Arabic Instruct** | Arabic translation of Alpaca instruction dataset | [Hugging Face](https://huggingface.co/datasets/Yasbok/Alpaca_arabic_instruct) |
| **Arabic Tweets** | 41GB+ of Arabic tweets (~4B words) | [Hugging Face](https://huggingface.co/datasets/pain/Arabic-Tweets) |
| **Rasaif** | Classical Arabic-English parallel texts (24 books) | [Hugging Face](https://huggingface.co/datasets/ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts) |
| **Shifaa Medical** | Arabic medical consultation dataset | [Hugging Face](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations) |
| **Shifaa Mental Health** | Arabic mental health consultation dataset | [Hugging Face](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations) |
| **Arabic Reasoning Dataset** | 9.2K Arabic instruction-based reasoning QA pairs | [Hugging Face](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset) |
| **Arabic Billion Words** | Abu El-Khair corpus: 5M+ newspaper articles, 1.5B+ words | [Hugging Face](https://huggingface.co/datasets/MohamedRashad/arabic-billion-words) |

### Speech Datasets

| Dataset | Description | Link |
|---|---|---|
| **ClArTTS** | Classical Arabic TTS dataset | [Hugging Face](https://huggingface.co/datasets/MBZUAI/ClArTTS) |
| **MADIS5** | Spoken Arabic dialects dataset | [Hugging Face](https://huggingface.co/datasets/badrex/MADIS5-spoken-arabic-dialects) |
| **SADA22** | MSA and Khaliji speech datasets | [Hugging Face](https://huggingface.co/datasets/badrex/arabic-speech-SADA22-MSA) |
| **Arabic Spontaneous Dialogue Dataset** | 300+ hours of real conversations | [Defined.ai](https://defined.ai/datasets/arabic-spontaneous-dialogue) |
| **Arabic Speech Corpus** | South Levantine Arabic (Damascian accent) speech corpus | [Hugging Face](https://huggingface.co/datasets/halabi2016/arabic_speech_corpus) |
| **Arabic-English Code-Switching** | Code-switching speech from YouTube and other sources | [Hugging Face](https://huggingface.co/datasets/MohamedRashad/arabic-english-code-switching) |
| **Egyptian Arabic ASR Clean** | ~72 hours of Egyptian Arabic speech aligned to text | [Hugging Face](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) |
| **SawtArabi** | Arabic speech dataset | [Hugging Face](https://huggingface.co/datasets/ArabicSpeech/sawtarabi) |

### Vision & Multimodal Datasets

| Dataset | Description | Link |
|---|---|---|
| **PEARL** | Multimodal Culturally-Aware Arabic Instruction Dataset | [Hugging Face](https://huggingface.co/datasets/UBC-NLP/PEARL) |
| **Arabic-Image-Captioning_100M** | 100 million Arabic image captions | [Hugging Face](https://huggingface.co/datasets/Misraj/Arabic-Image-Captioning_100M) |
| **Calliar** | Online Arabic calligraphy dataset (2500 samples) | [GitHub](https://github.com/ARBML/Calliar) |
| **arabic-img2md** | 15K PDF pages paired with Markdown for Arabic OCR | [Hugging Face](https://huggingface.co/datasets/MohamedRashad/arabic-img2md) |
| **Arabic-OCR-Dataset** | Comprehensive Arabic OCR dataset (1M+ samples) | [Hugging Face](https://huggingface.co/datasets/mssqpi/Arabic-OCR-Dataset) |
| **Arabic-VLM-Full-Pearl** | 309K multimodal examples for Arabic VLM training | [Hugging Face](https://huggingface.co/datasets/MohamedRashad/Arabic-VLM-Full-Pearl) |

## Essential Tools & Libraries

### Toolkits & Preprocessing

| Tool | Description | Link |
|---|---|---|
| **camel_tools** | Suite of Arabic NLP tools (morphology, POS, NER, etc.) | [GitHub](https://github.com/CAMeL-Lab/camel_tools) |
| **Farasa** | Fast and accurate Arabic text processing toolkit | [Website](https://farasa.qcri.org/) |
| **SinaTools** | Open source toolkit by SinaLab (Python APIs, CLI) | [GitHub](https://github.com/SinaLab/sinatools) |
| **Qalsadi** | Arabic morphological analyzer and lemmatizer | [GitHub](https://github.com/linuxscout/qalsadi) |
| **PyArabic** | Python package for basic Arabic text manipulation | [GitHub](https://github.com/linuxscout/pyarabic) |
| **tnkeeh** | Arabic text cleaning, normalization, and preprocessing | [GitHub](https://github.com/ARBML/tnkeeh) |
| **arabicprocess** | Python library for Arabic text preprocessing | [PyPI](https://pypi.org/project/arabicprocess/) |
| **MADAMIRA** | Morphological analysis, diacritization, POS tagging | [Website](https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/computational-approaches-to-modeling-language-lab/research/morphological-analysis-of-arabic.html) |
| **Maha** | Text processing library specially developed for Arabic text | [GitHub](https://github.com/TRoboto/Maha) |
| **Mishkal** | Arabic text diacritizer (rule-based) | [GitHub](https://github.com/linuxscout/mishkal) |

### Specialized Libraries

| Library | Task | Link |
|---|---|---|
| **klaam** | Speech Recognition, Classification, TTS | [GitHub](https://github.com/ARBML/klaam) |
| **tkseem** | Arabic Tokenization | [GitHub](https://github.com/ARBML/tkseem) |
| **arabic-stop-words** | Largest list of Arabic stop words | [GitHub](https://github.com/mohataher/arabic-stop-words) |
| **qawafi** | Arabic poetry analysis | [GitHub](https://github.com/ARBML/qawafi) |
| **arabic_vocalizer** | Arabic deep-learning diacritization models (ONNX format) | [GitHub](https://github.com/nipponjo/arabic_vocalizer) |

### Translation Tools

| Tool | Description | Link |
|---|---|---|
| **Helsinki-NLP/opus-mt-en-ar** | English to Arabic neural machine translation | [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) |
| **Helsinki-NLP/opus-mt-ar-en** | Arabic to English neural machine translation (12.4M+ downloads) | [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en) |

## Top Research Papers & Conferences

### Conferences & Workshops

| Conference | Year | Link |
|---|---|---|
| **ArabicNLP 2025** | 2025 | [Website](https://arabicnlp2025.sigarab.org/) |
| **ArabicNLP 2024** | 2024 | [ACL Anthology](https://aclanthology.org/events/arabicnlp-2024/) |
| **ArabicNLP 2023** | 2023 | [ACL Anthology](https://aclanthology.org/venues/arabicnlp/) |
| **Arabic NLP Winter School** | 2025 | [Website](https://winterschool2025.sigarab.org/) |
| **AbjadNLP Workshop** | 2025-2026 | [Website](https://wp.lancs.ac.uk/abjad/) |
| **OSACT (Open-Source Arabic Corpora and Tools)** | Ongoing | [Website](https://osact-lrec.github.io/) |

### Foundational & Survey Papers

1.  **The Landscape of Arabic Large Language Models** (2025) [[1]](#references)
2.  **AraBERT: Transformer-based Model for Arabic Language Understanding** (Antoun et al., 2020) [[2]](#references)
3.  **The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models (CAMeLBERT)** (Inoue et al., 2021) [[3]](#references)
4.  **ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic** (Abdul-Mageed et al., 2021) [[4]](#references)
5.  **Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned Open Generative Large Language Models** (Sengupta et al., 2023) [[5]](#references)
6.  **Wojood: Nested Arabic Named Entity Corpus and Recognition** (Jarrar et al., 2022) [[6]](#references)
7.  **Deep Learning for Arabic NLP: A Survey** (Al-Ayyoub et al., 2018) [[7]](#references)
8.  **Evaluating Arabic Large Language Models: A Survey of Benchmarks, Methods, and Gaps** (Alzubaidi et al., 2025) [[8]](#references)

### Recent Papers (2024-2025)

*   **Swan and ArabicMTEB: Dialect-Aware, Arabic-Centric, Cross-Lingual Language Understanding** (Bhatia et al., 2024) [[9]](#references)
*   **GATE: General Arabic Text Embedding for Enhanced Semantic Textual Similarity** (Nacar et al., 2025) [[10]](#references)
*   **A Survey of Large Language Models for Arabic Language and its Dialects** (Mashaabi et al., 2024) [[11]](#references)
*   **Hate speech detection in the Arabic language: corpus design, construction, and evaluation** (2024) [[12]](#references)
*   **NADI 2024: The Fifth Nuanced Arabic Dialect Identification Shared Task** (2024) [[13]](#references)
*   **CATT: Character-based Arabic Tashkeel Transformer** (2024) [[14]](#references)
*   **ArabianGPT: Native Arabic GPT-based Large Language Model** (2024) [[15]](#references)
*   **SambaLingo: Teaching Large Language Models New Languages** (2024) [[16]](#references)

## Tutorials & Learning Resources

### Academic Programs

| Resource | Description | Link |
|---|---|---|
| **Arabic NLP Winter School** | Two-day intensive program at MBZUAI covering Arabic NLP research and development (Jan 2025) | [Website](https://winterschool2025.sigarab.org/) |
| **ArabicNLP Conference** | Annual ACL-affiliated conference by SIGARAB for Arabic NLP research | [Website](https://arabicnlp2025.sigarab.org/) |
| **AbjadNLP Workshop** | Workshop on NLP for languages using Arabic script | [Website](https://wp.lancs.ac.uk/abjad/) |

### Online Resources

| Resource | Description | Link |
|---|---|---|
| **Hugging Face NLP Course** | Free NLP course covering transformers, applicable to Arabic models | [Website](https://huggingface.co/learn/nlp-course) |
| **AUB MIND Lab Arabic-NLP Demo** | Interactive demo for Arabic NLP tasks (NER, sentiment, etc.) | [Hugging Face Space](https://huggingface.co/spaces/aubmindlab/Arabic-NLP) |
| **MoroccoAI Darija NLP Resources** | Curated resources for Moroccan Arabic dialect NLP | [GitHub](https://github.com/MoroccoAI/Arabic-Darija-NLP-Resources) |
| **NNLP-IL Arabic Resources** | Comprehensive list of Arabic NLP resources | [GitHub](https://github.com/NNLP-IL/Arabic-Resources) |

## Companies & Startups

Companies and organizations building Arabic AI/NLP products and services.

| Company | Country | Focus | Notable Products |
|---|---|---|---|
| **G42** | UAE | AI holding company, Arabic LLMs | Jais LLM, enterprise AI |
| **Inception AI** | UAE | Arabic-centric foundation models | Jais model family (with Cerebras) |
| **SILMA AI** | - | Arabic-first LLMs and benchmarks | SILMA LLMs, Arabic Broad Benchmark |
| **Technology Innovation Institute (TII)** | UAE | Open-source LLMs | Falcon LLM family |
| **Saal.ai** | UAE | Cognitive AI solutions | Arabic NLP, speech processing, generative AI |
| **Velents** | Egypt | Enterprise Arabic AI | Agent.sa (Arabic-speaking AI employee) |
| **Cohere** | Canada | Multilingual LLMs | Command R Arabic, RAG optimization |
| **Mistral AI** | France | Multilingual LLMs | Mistral Saba (Arabic-optimized) |
| **SambaNova Systems** | USA | Arabic language adaptation | SambaLingo Arabic models |

## Awesome Lists

- [01walid/awesome-arabic](https://github.com/01walid/awesome-arabic) - A curated list of awesome projects, libraries, and resources for the Arabic language.
- [Curated-Awesome-Lists/awesome-arabic-nlp](https://github.com/Curated-Awesome-Lists/awesome-arabic-nlp) - A comprehensive list of Arabic NLP resources.
- [MoroccoAI/Arabic-Darija-NLP-Resources](https://github.com/MoroccoAI/Arabic-Darija-NLP-Resources) - Resources for Moroccan Arabic dialect NLP.
- [NNLP-IL/Arabic-Resources](https://github.com/NNLP-IL/Arabic-Resources) - A comprehensive list of Arabic NLP resources.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to add new resources or update existing ones.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

## References

[1] The Landscape of Arabic Large Language Models. (2025). *arXiv preprint arXiv:2506.01340*. https://arxiv.org/html/2506.01340v1

[2] Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding. In *Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT)*. https://aclanthology.org/2020.osact-1.2/

[3] Inoue, G., et al. (2021). The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models. In *Proceedings of the Sixth Arabic Natural Language Processing Workshop*. https://aclanthology.org/2021.wanlp-1.10/

[4] Abdul-Mageed, M., et al. (2021). ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*. https://aclanthology.org/2021.acl-long.551/

[5] Sengupta, N., et al. (2023). Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned Open Generative Large Language Models. *arXiv preprint arXiv:2308.16149*. https://arxiv.org/abs/2308.16149

[6] Jarrar, M., et al. (2022). Wojood: Nested Arabic Named Entity Corpus and Recognition using BERT. *arXiv preprint arXiv:2205.09651*. https://arxiv.org/abs/2205.09651

[7] Al-Ayyoub, M., et al. (2018). Deep learning for Arabic NLP: A survey. *Journal of Computational Science*, 26, 522-531. https://www.sciencedirect.com/science/article/pii/S1877750317303757

[8] Alzubaidi, A., et al. (2025). Evaluating Arabic Large Language Models: A Survey of Benchmarks, Methods, and Gaps. *arXiv preprint arXiv:2510.13430*. https://arxiv.org/abs/2510.13430

[9] Bhatia, G., et al. (2024). Swan and ArabicMTEB: Dialect-Aware, Arabic-Centric, Cross-Lingual Language Understanding. *arXiv preprint arXiv:2411.01192*. https://arxiv.org/abs/2411.01192

[10] Nacar, O., et al. (2025). GATE: General Arabic Text Embedding for Enhanced Semantic Textual Similarity. *arXiv preprint arXiv:2505.24581*. https://arxiv.org/abs/2505.24581

[11] Mashaabi, M., et al. (2024). A Survey of Large Language Models for Arabic Language and its Dialects. *arXiv preprint arXiv:2410.20238*. https://arxiv.org/abs/2410.20238

[12] Hate speech detection in the Arabic language: corpus design, construction, and evaluation. (2024). *Frontiers in Artificial Intelligence*. https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1345445/full

[13] NADI 2024: The Fifth Nuanced Arabic Dialect Identification Shared Task. (2024). In *Proceedings of the Second Arabic Natural Language Processing Conference*. https://arxiv.org/abs/2407.04910

[14] CATT: Character-based Arabic Tashkeel Transformer. (2024). *arXiv preprint arXiv:2407.03236*. https://arxiv.org/abs/2407.03236

[15] ArabianGPT: Native Arabic GPT-based Large Language Model. (2024). *arXiv preprint arXiv:2402.15313*. https://arxiv.org/abs/2402.15313

[16] SambaLingo: Teaching Large Language Models New Languages. (2024). *arXiv preprint arXiv:2404.05829*. https://arxiv.org/abs/2404.05829
