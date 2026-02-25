<div align="center">

<img src="assets/banner.svg" alt="Awesome Arabic NLP Banner" width="900"/>

<br/>
<br/>

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Stars](https://img.shields.io/github/stars/h9-tec/Awesome_Arabic_NLP?style=social)](https://github.com/h9-tec/Awesome_Arabic_NLP)

**The most comprehensive, up-to-date collection of Arabic NLP resources on the internet.**<br/>
Models · Datasets · Tools · Research · Companies

</div>

---

## 📑 Table of Contents

<table>
<tr>
<td width="50%" valign="top">

- [🏢 Key Organizations](#-key-organizations)
- [🏆 Benchmarks & Leaderboards](#-benchmarks--leaderboards)
- [🤖 State-of-the-Art Models](#-state-of-the-art-models)
  - [💬 Large Language Models](#-large-language-models-llms)
  - [🌐 Multimodal Models](#-multimodal-models)
  - [🧠 Transformer-based Models](#-transformer-based-models)
  - [🔤 Embedding Models](#-embedding-models)
  - [🎯 Task-Specific Models](#-task-specific-models)
- [🎙️ Audio Models](#%EF%B8%8F-audio-models)
- [👁️ Vision Models](#%EF%B8%8F-vision-models)

</td>
<td width="50%" valign="top">

- [✏️ Diacritization (Tashkeel)](#%EF%B8%8F-diacritization-tashkeel)
- [🗣️ Dialect Identification](#%EF%B8%8F-dialect-identification)
- [📊 Key Datasets](#-key-datasets)
- [🔧 Essential Tools & Libraries](#-essential-tools--libraries)
- [📄 Research Papers & Conferences](#-research-papers--conferences)
- [🎓 Tutorials & Learning Resources](#-tutorials--learning-resources)
- [🏭 Companies & Startups](#-companies--startups)
- [📚 Awesome Lists](#-awesome-lists)

</td>
</tr>
</table>

---

## 🏢 Key Organizations

> Research labs and institutions driving Arabic NLP forward.

| Organization | Focus | Key Contributions | Link |
|:---|:---|:---|:---:|
| **AUB MIND Lab** | Foundational Arabic NLP models | AraBERT, AraGPT2, AraELECTRA | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/aub-mind) |
| **CAMeL Lab, NYUAD** | Arabic NLP tools and models | CAMeLBERT, camel_tools | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/CAMeL-Lab) |
| **UBC-NLP** | Dialectal Arabic, multimodal models | MARBERT, AraT5, NileChat, PEARL, Dallah | [![HF](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/UBC-NLP) |
| **QCRI** | Arabic LLMs, text processing | Fanar LLMs, AraDiCE, Farasa | [![HF](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/QCRI) |
| **SILMA AI** | State-of-the-art Arabic LLMs | SILMA LLMs, Arabic Broad Benchmark | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://silma.ai/) |
| **MBZUAI** | Multimodal and speech models | AIN, ArTST, ClArTTS | [![HF](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/MBZUAI) |
| **ARBML** | Democratizing Arabic NLP | masader, klaam, tkseem | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/ARBML) |
| **NAMAA-Space** | OCR and Egyptian Arabic models | Qari-OCR, EgypTalk-ASR | [![HF](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/NAMAA-Space) |
| **Omartificial-Intelligence-Space** | Arabic embedding models | GATE, Matryoshka embeddings | [![HF](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Omartificial-Intelligence-Space) |
| **TII (Technology Innovation Institute)** | Arabic LLM benchmarks, Falcon | Open Arabic LLM Leaderboard, Falcon LLM | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://www.tii.ae/) |
| **SDAIA (Saudi Data & AI Authority)** | Sovereign Arabic LLM | ALLaM model | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://sdaia.gov.sa/) |
| **SinaLab, Birzeit University** | Arabic NLP tools and datasets | SinaTools, Wojood NER | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/SinaLab) |
| **G42 / Inception AI** | Arabic-centric LLMs | Jais LLM family | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://www.g42.ai/) |
| **FreedomIntelligence** | Arabic LLMs and alignment | AceGPT, Arabic cultural datasets | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/FreedomIntelligence) |
| **Helsinki-NLP** | Machine translation models | OPUS-MT Arabic translation models | [![HF](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Helsinki-NLP) |
| **LightOn AI** | Arabic web data | ArabicWeb24 corpus (39B+ tokens) | [![HF](https://img.shields.io/badge/-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/lightonai) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 🏆 Benchmarks & Leaderboards

> Standardized evaluation frameworks for Arabic language models.

| Benchmark | Description | Link |
|:---|:---|:---:|
| **MTEB Arabic Leaderboard** | Massive Text Embedding Benchmark for Arabic | [![HF](https://img.shields.io/badge/-Leaderboard-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mteb/leaderboard) |
| **Arabic Broad Leaderboard (ABL)** | NextGen evaluation for Arabic LLMs by SILMA AI | [![HF](https://img.shields.io/badge/-Leaderboard-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/silma-ai/Arabic-LLM-Broad-Leaderboard) |
| **Open Arabic LLM Leaderboard** | Evaluation of Arabic LLMs across multiple benchmarks | [![HF](https://img.shields.io/badge/-Leaderboard-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard) |
| **ALUE** | Arabic Language Understanding Evaluation benchmark | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://www.alue.org/) |
| **BALSAM** | Benchmark of Arabic Language AI Systems and Models | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://benchmarks.ksaa.gov.sa/) |
| **SILMA RAGQA Benchmark** | Evaluates Arabic/English LMs in Extractive QA tasks | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/silma-ai/silma-rag-qa-benchmark-v1.0) |
| **Arabic Broad Benchmark (ABB)** | Comprehensive evaluation tool for Arabic LLMs | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/silma-ai/arabic-broad-benchmark) |
| **ArabicMMLU** | Multi-task language understanding from school exams | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) |
| **GATmath and GATLc** | Benchmarks from Saudi GAT exams | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0329129) |
| **ArabicRAGB** | Arabic RAG Benchmark (multi-dialect) | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/HeshamHaroon/ArabicRAGB) |
| **ACVA** | Arabic Cultural Value Alignment (8000+ questions, 58 areas) | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 🤖 State-of-the-Art Models

### 💬 Large Language Models (LLMs)

> Arabic-centric and multilingual LLMs with strong Arabic capabilities.

| Model | Params | Developer | Key Features | Link |
|:---|:---:|:---|:---|:---:|
| **Jais** | 13B, 30B | Inception AI, Cerebras | Arabic-centric, bilingual, instruction-tuned | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/inceptionai/jais-30b-v3) |
| **SILMA 1.0** | 9B | SILMA AI | Top-ranked Arabic LLM built on Gemma | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/silma-ai/SILMA-9B-Instruct-v1.0) |
| **ALLaM** | 7B | SDAIA & IBM | Saudi's sovereign model, enterprise-focused | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview) |
| **Fanar-1-9B** | 9B | QCRI | Arabic-English LLM | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/QCRI/Fanar-1-9B-Instruct) |
| **AceGPT** | 7B | FreedomIntelligence | Top performance, culturally aligned | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/FreedomIntelligence/AceGPT) |
| **Atlas-Chat** | 2B-27B | MBZUAI-Paris Lab | Moroccan Darija dialect | [![HF](https://img.shields.io/badge/-Collection-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/collections/MBZUAI-Paris/atlas-chat) |
| **NileChat-3B** | 3B | UBC-NLP | Egyptian and Moroccan dialects | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/UBC-NLP/NileChat-3B) |
| **Nile-Chat** | 4B-12B | MBZUAI-Paris Lab | Egyptian Arabic and Arabizi scripts | [![HF](https://img.shields.io/badge/-Collection-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/collections/MBZUAI-Paris/nile-chat) |
| **AraGPT2** | 1.5B | AUB MIND Lab | GPT-2 for Arabic text generation | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/aubmindlab/aragpt2-mega) |
| **Command R7B Arabic** | 7B | Cohere | Arabic-optimized Command R variant | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/CohereLabs/c4ai-command-r7b-arabic-02-2025) |
| **SambaLingo-Arabic** | 7B, 70B | SambaNova | Arabic-adapted Llama 2 | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/sambanovasystems/SambaLingo-Arabic-Chat) |
| **ArabianGPT** | - | - | Native Arabic GPT-based LLM | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2402.15313) |
| **Llama 3.3** | 70B | Meta | Strong Arabic performance | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| **Qwen 3** | 0.6B-235B | Alibaba | Multilingual with Arabic support | [![HF](https://img.shields.io/badge/-Collection-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) |
| **Gemma 3** | 1B-27B | Google | Multimodal capabilities | [![HF](https://img.shields.io/badge/-Collection-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) |
| **Cohere Command-A** | 111B | Cohere | Optimized for RAG | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/CohereForAI/c4ai-command-a-03-2025) |
| **Mistral Saba** | 24B | Mistral | Commercial API | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://mistral.ai/news/mistral-saba) |

### 🌐 Multimodal Models

| Model | Params | Developer | Key Features | Link |
|:---|:---:|:---|:---|:---:|
| **AIN** | 8B | MBZUAI | Arabic-centric Large Multimodal Model | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/mbzuai-oryx/AIN) |
| **Dallah** | - | UBC-NLP | Advanced multimodal LLM for Arabic | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/UBC-NLP/dallah) |

### 🧠 Transformer-based Models

| Model | Developer | Key Features | Link |
|:---|:---|:---|:---:|
| **AraBERT** | AUB MIND Lab | First BERT for Arabic, multiple versions | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/aub-mind/arabert) |
| **AraBERTv02** | AUB MIND Lab | Improved tokenization (135M params, 12M+ downloads) | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/aubmindlab/bert-base-arabertv02) |
| **CAMeLBERT** | CAMeL Lab, NYUAD | MSA, Dialectal, and Classical Arabic | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/CAMeL-Lab/CAMeLBERT) |
| **MARBERT** | UBC-NLP | Focused on Dialectal Arabic and MSA | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/UBC-NLP/marbert) |
| **MARBERTv2** | UBC-NLP | Updated with improved dialectal coverage | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/UBC-NLP/MARBERTv2) |
| **AraELECTRA** | AUB MIND Lab | ELECTRA for Arabic | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/aubmindlab/araelectra-base-discriminator) |
| **AraT5** | UBC-NLP | T5 for Arabic summarization, translation, paraphrasing | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/UBC-NLP/AraT5-base) |

### 🔤 Embedding Models

| Model | Developer | Key Features | Link |
|:---|:---|:---|:---:|
| **GATE-AraBert-v1** | Omartificial-Intelligence-Space | SOTA on MTEB Arabic STS | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Omartificial-Intelligence-Space/GATE-AraBert-v1) |
| **Arabic-Triplet-Matryoshka-V2** | Omartificial-Intelligence-Space | Matryoshka representation | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2) |
| **Swan** | UBC-NLP | Dialect-aware, cross-lingual | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2411.01192) |
| **asafaya/bert-base-arabic** | asafaya | BERT-based Arabic embeddings | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/asafaya/bert-base-arabic) |
| **ModernBERT-Arabic** | BounharAbdelaziz | ModernBERT-based sentence embeddings | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/BounharAbdelaziz/ModernBERT-Arabic-Embeddings) |
| **DIMI-embedding** | AhmedZaky1 | Matryoshka + AraBERT for NLI | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/AhmedZaky1/DIMI-embedding-matryoshka-arabic) |

### 🎯 Task-Specific Models

| Model | Task | Key Features | Link |
|:---|:---|:---|:---:|
| **CAMeLBERT-MSA-Sentiment** | Sentiment Analysis | Fine-tuned for MSA sentiment | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment) |
| **t5-arabic-summarization** | Summarization | T5 for Arabic news summarization | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/malmarjeh/t5-arabic-text-summarization) |
| **opus-mt-en-ar** | Translation EN→AR | Helsinki-NLP (3.5M+ downloads) | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) |
| **opus-mt-ar-en** | Translation AR→EN | Helsinki-NLP (12.4M+ downloads) | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en) |
| **arabic-gec-v1** | Grammar Correction | Gemma-3-1b for Arabic GEC | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/alnnahwi/gemma-3-1b-arabic-gec-v1) |
| **Arabic-Text-Correction** | Text Correction | AraT5-based text correction | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/SuperSl6/Arabic-Text-Correction) |
| **arat5-dialects-translation** | Dialect→MSA | AraT5 dialect translation | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/PRAli22/arat5-arabic-dialects-translation) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 🎙️ Audio Models

### 🗣️ Speech Recognition (ASR)

| Model | Key Features | Link |
|:---|:---|:---:|
| **openai/whisper-large-v3** | Supports Arabic among many languages | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/openai/whisper-large-v3) |
| **MasriSwitch-Gemma3n** | Egyptian Arabic code-switching transcription | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/oddadmix/MasriSwitch-Gemma3n-Transcriber-v1) |
| **wav2vec2-large-xlsr-53-arabic** | Fine-tuned on Common Voice & Arabic Speech Corpus | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic) |
| **artst_asr_v3** | ArTST for ASR on MGB2 (best for MSA) | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/MBZUAI/artst_asr_v3) |
| **EgypTalk-ASR-v2** | High-performance ASR for Egyptian Arabic | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/NAMAA-Space/EgypTalk-ASR-v2) |

### 🔊 Text-to-Speech (TTS)

| Model | Key Features | Link |
|:---|:---|:---:|
| **facebook/mms-tts-ara** | Facebook's Massively Multilingual Speech | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/facebook/mms-tts-ara) |
| **speecht5_tts_clartts_ar** | SpeechT5 for Classical Arabic TTS | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar) |
| **F5-TTS-Arabic** | F5-TTS with regional diversity | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/IbrahimSalah/F5-TTS-Arabic) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 👁️ Vision Models

### 📖 Optical Character Recognition (OCR)

| Model | Key Features | Link |
|:---|:---|:---:|
| **Qari-OCR-0.1-VL-2B-Instruct** | Qwen2 VL 2B fine-tuned for Arabic OCR | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct) |
| **arabic-large-nougat** | End-to-end structured OCR for Arabic | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/MohamedRashad/arabic-large-nougat) |

### 🖼️ Image Captioning

| Model | Key Features | Link |
|:---|:---|:---:|
| **blip-Arabic-flickr-8k** | BLIP fine-tuned for Arabic image captioning | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/omarsabri8756/blip-Arabic-flickr-8k) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## ✏️ Diacritization (Tashkeel)

> Models and tools for adding harakat (حركات) to Arabic text.

### Models

| Model / System | Key Features | Link |
|:---|:---|:---:|
| **CATT** | Character-based Tashkeel Transformer, SOTA results | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.03236) |
| **Fine-Tashkeel** | Fine-tuned ByT5, 40% WER reduction | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://www.researchgate.net/publication/372616004) |
| **Sadeed** | Small language model for diacritization | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.21635) |
| **Shakkala** | Neural vocalization using bidirectional LSTM | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/AliOsm/shakkelha) |
| **Mishkal** | Rule-based diacritizer with dictionary lookups | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/linuxscout/mishkal) |

### Datasets

| Dataset | Description | Link |
|:---|:---|:---:|
| **Tashkeela** | Arabic diacritization corpus | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/Anwarvic/Arabic-Tashkeela-Model) |
| **arabic-text-diacritization** | Benchmark dataset with systems comparison | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/AliOsm/arabic-text-diacritization) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 🗣️ Dialect Identification

> Resources for identifying and classifying Arabic dialects.

### Shared Tasks

| Task | Description | Link |
|:---|:---|:---:|
| **NADI 2025** | Multidialectal Arabic Speech Processing (8-way dialect + ASR) | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://nadi.dlnlp.ai/2025/) |
| **NADI 2024** | Fifth Nuanced Arabic Dialect Identification | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.04910) |
| **NADI Shared Tasks** | Ongoing series of Arabic DID shared tasks | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://nadi.dlnlp.ai/) |

### Datasets

| Dataset | Description | Link |
|:---|:---|:---:|
| **QADI** | Twitter-based multi-class dialect classification | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Abdelrahman-Rezk/Arabic_Dialect_Identification) |
| **Arabic POS Dialect** | POS tagging in Arabic dialects | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/QCRI/arabic_pos_dialect) |
| **Arabic Dialects to MSA** | Parallel dialect-MSA corpus | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/PRAli22/Arabic_dialects_to_MSA) |
| **Casablanca** | Multidialectal Arabic speech dataset (NADI 2025) | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.04527) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 📊 Key Datasets

### 📝 Text Datasets

| Dataset | Description | Link |
|:---|:---|:---:|
| **masader** | Largest public catalogue of Arabic NLP datasets (600+) | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/ARBML/masader) |
| **101 Billion Arabic Words** | Massive Arabic web corpus | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset) |
| **ArabicWeb24** | 39B+ tokens of high-quality Arabic web content | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/lightonai/ArabicWeb24) |
| **ArabicText-Large** | 743K articles for LLM training | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Jr23xd23/ArabicText-Large) |
| **Arabic Billion Words** | Abu El-Khair corpus: 5M+ articles, 1.5B+ words | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/MohamedRashad/arabic-billion-words) |
| **Arabic Tweets** | 41GB+ of Arabic tweets (~4B words) | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/pain/Arabic-Tweets) |
| **Wojood** | Nested NER corpus (550K tokens) | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/SinaLab/ArabicNER) |
| **CIDAR** | Culturally relevant instruction dataset (10K pairs) | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/arbml/CIDAR) |
| **Arabic_Function_Calling** | First Arabic function calling dataset (50K+ samples) | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/HeshamHaroon/Arabic_Function_Calling) |
| **ArabicaQA** | Large-scale Arabic Question Answering | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/DataScienceUIBK/ArabicaQA) |
| **Mixed Arabic Datasets (MAD)** | Community-driven diverse Arabic texts | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/M-A-D/Mixed-Arabic-Datasets-Repo) |
| **Arabic-OpenHermes-2.5** | Arabic OpenHermes instruction dataset | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/2A2I/Arabic-OpenHermes-2.5) |
| **Alpaca Arabic Instruct** | Arabic Alpaca instruction dataset | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Yasbok/Alpaca_arabic_instruct) |
| **Rasaif** | Classical Arabic-English parallel texts (24 books) | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts) |
| **Shifaa Medical** | Arabic medical consultation dataset | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations) |
| **Shifaa Mental Health** | Arabic mental health consultations | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations) |
| **Arabic Reasoning Dataset** | 9.2K instruction-based reasoning QA pairs | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset) |
| **Gazelle** | Arabic writing assistance dataset | [![Paper](https://img.shields.io/badge/-Paper-B31B1B?logo=arxiv&logoColor=white)](https://huggingface.co/papers/2410.18163) |
| **arabic-hate-speech-superset** | Comprehensive hate speech detection | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/manueltonneau/arabic-hate-speech-superset) |
| **ArabicCorpus2B** | 1.9B word Arabic corpus | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/tarekeldeeb/ArabicCorpus2B) |
| **The Arabic E-Book Corpus** | 1,745 books (81.5M words) | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/mohres/The_Arabic_E-Book_Corpus) |
| **BAREC Corpus** | Arabic Readability Assessment | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0) |
| **palm** | Human-created Arabic instruction dataset | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/UBC-NLP/palm) |
| **dialogue-arabic-dialects** | Levantine, Egyptian, Gulf dialect dialogues | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/tareknaous/dialogue-arabic-dialects) |

### 🎤 Speech Datasets

| Dataset | Description | Link |
|:---|:---|:---:|
| **ClArTTS** | Classical Arabic TTS dataset | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/MBZUAI/ClArTTS) |
| **Arabic Speech Corpus** | South Levantine Arabic (Damascian accent) | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/halabi2016/arabic_speech_corpus) |
| **Arabic-English Code-Switching** | Code-switching speech from YouTube | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/MohamedRashad/arabic-english-code-switching) |
| **Egyptian Arabic ASR Clean** | ~72 hours of Egyptian Arabic speech | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) |
| **MADIS5** | Spoken Arabic dialects | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/badrex/MADIS5-spoken-arabic-dialects) |
| **SADA22** | MSA and Khaliji speech | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/badrex/arabic-speech-SADA22-MSA) |
| **SawtArabi** | Arabic speech dataset | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ArabicSpeech/sawtarabi) |

### 🖼️ Vision & Multimodal Datasets

| Dataset | Description | Link |
|:---|:---|:---:|
| **PEARL** | Multimodal Culturally-Aware Arabic Instruction Dataset | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/UBC-NLP/PEARL) |
| **Arabic-Image-Captioning_100M** | 100 million Arabic image captions | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Misraj/Arabic-Image-Captioning_100M) |
| **Calliar** | Online Arabic calligraphy (2500 samples) | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/ARBML/Calliar) |
| **arabic-img2md** | 15K PDF pages paired with Markdown | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/MohamedRashad/arabic-img2md) |
| **Arabic-OCR-Dataset** | 1M+ Arabic OCR samples | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/mssqpi/Arabic-OCR-Dataset) |
| **Arabic-VLM-Full-Pearl** | 309K multimodal examples for VLM training | [![HF](https://img.shields.io/badge/-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/MohamedRashad/Arabic-VLM-Full-Pearl) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 🔧 Essential Tools & Libraries

### ⚙️ Toolkits & Preprocessing

| Tool | Description | Link |
|:---|:---|:---:|
| **camel_tools** | Suite of Arabic NLP tools (morphology, POS, NER, etc.) | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/CAMeL-Lab/camel_tools) |
| **Farasa** | Fast and accurate Arabic text processing toolkit | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://farasa.qcri.org/) |
| **SinaTools** | Open source toolkit by SinaLab (Python APIs, CLI) | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/SinaLab/sinatools) |
| **Qalsadi** | Arabic morphological analyzer and lemmatizer | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/linuxscout/qalsadi) |
| **PyArabic** | Python package for Arabic text manipulation | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/linuxscout/pyarabic) |
| **tnkeeh** | Arabic text cleaning, normalization, preprocessing | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/ARBML/tnkeeh) |
| **Maha** | Text processing library for Arabic text | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/TRoboto/Maha) |
| **Mishkal** | Arabic text diacritizer (rule-based) | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/linuxscout/mishkal) |
| **arabicprocess** | Python library for Arabic preprocessing | [![PyPI](https://img.shields.io/badge/-PyPI-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/arabicprocess/) |
| **MADAMIRA** | Morphological analysis, diacritization, POS tagging | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/computational-approaches-to-modeling-language-lab/research/morphological-analysis-of-arabic.html) |

### 📚 Specialized Libraries

| Library | Task | Link |
|:---|:---|:---:|
| **klaam** | Speech Recognition, Classification, TTS | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/ARBML/klaam) |
| **tkseem** | Arabic Tokenization | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/ARBML/tkseem) |
| **arabic-stop-words** | Largest list of Arabic stop words | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/mohataher/arabic-stop-words) |
| **qawafi** | Arabic poetry analysis | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/ARBML/qawafi) |
| **arabic_vocalizer** | Deep-learning diacritization (ONNX format) | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/nipponjo/arabic_vocalizer) |

### 🌍 Translation

| Tool | Description | Link |
|:---|:---|:---:|
| **opus-mt-en-ar** | English → Arabic neural MT | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar) |
| **opus-mt-ar-en** | Arabic → English neural MT (12.4M+ downloads) | [![HF](https://img.shields.io/badge/-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 📄 Research Papers & Conferences

### 🎤 Conferences & Workshops

| Conference | Year | Link |
|:---|:---:|:---:|
| **ArabicNLP 2025** | 2025 | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://arabicnlp2025.sigarab.org/) |
| **ArabicNLP 2024** | 2024 | [![ACL](https://img.shields.io/badge/-ACL_Anthology-D92D2D)](https://aclanthology.org/events/arabicnlp-2024/) |
| **ArabicNLP 2023** | 2023 | [![ACL](https://img.shields.io/badge/-ACL_Anthology-D92D2D)](https://aclanthology.org/venues/arabicnlp/) |
| **Arabic NLP Winter School** | 2025 | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://winterschool2025.sigarab.org/) |
| **AbjadNLP Workshop** | 2025-2026 | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://wp.lancs.ac.uk/abjad/) |
| **OSACT** | Ongoing | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://osact-lrec.github.io/) |

### 📖 Foundational & Survey Papers

1. **The Landscape of Arabic Large Language Models** (2025) [[1]](#references)
2. **AraBERT: Transformer-based Model for Arabic Language Understanding** (Antoun et al., 2020) [[2]](#references)
3. **The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models** (Inoue et al., 2021) [[3]](#references)
4. **ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic** (Abdul-Mageed et al., 2021) [[4]](#references)
5. **Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned LLMs** (Sengupta et al., 2023) [[5]](#references)
6. **Wojood: Nested Arabic Named Entity Corpus and Recognition** (Jarrar et al., 2022) [[6]](#references)
7. **Deep Learning for Arabic NLP: A Survey** (Al-Ayyoub et al., 2018) [[7]](#references)
8. **Evaluating Arabic LLMs: A Survey of Benchmarks, Methods, and Gaps** (Alzubaidi et al., 2025) [[8]](#references)

### 🆕 Recent Papers (2024-2025)

- **Swan and ArabicMTEB** — Dialect-Aware, Cross-Lingual Language Understanding (Bhatia et al., 2024) [[9]](#references)
- **GATE** — General Arabic Text Embedding for Enhanced STS (Nacar et al., 2025) [[10]](#references)
- **A Survey of LLMs for Arabic Language and its Dialects** (Mashaabi et al., 2024) [[11]](#references)
- **Hate speech detection in Arabic** — corpus design and evaluation (2024) [[12]](#references)
- **NADI 2024** — Fifth Nuanced Arabic Dialect Identification Shared Task (2024) [[13]](#references)
- **CATT** — Character-based Arabic Tashkeel Transformer (2024) [[14]](#references)
- **ArabianGPT** — Native Arabic GPT-based LLM (2024) [[15]](#references)
- **SambaLingo** — Teaching LLMs New Languages (2024) [[16]](#references)

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 🎓 Tutorials & Learning Resources

### 🏫 Academic Programs

| Resource | Description | Link |
|:---|:---|:---:|
| **Arabic NLP Winter School** | Two-day intensive at MBZUAI (Jan 2025) | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://winterschool2025.sigarab.org/) |
| **ArabicNLP Conference** | Annual ACL-affiliated conference by SIGARAB | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://arabicnlp2025.sigarab.org/) |
| **AbjadNLP Workshop** | NLP for languages using Arabic script | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://wp.lancs.ac.uk/abjad/) |

### 💻 Online Resources

| Resource | Description | Link |
|:---|:---|:---:|
| **Hugging Face NLP Course** | Free NLP course (applicable to Arabic models) | [![HF](https://img.shields.io/badge/-Course-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/learn/nlp-course) |
| **AUB MIND Lab Arabic-NLP Demo** | Interactive demo for Arabic NLP tasks | [![HF](https://img.shields.io/badge/-Space-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/aubmindlab/Arabic-NLP) |
| **MoroccoAI Darija Resources** | Curated Moroccan Arabic dialect NLP resources | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/MoroccoAI/Arabic-Darija-NLP-Resources) |
| **NNLP-IL Arabic Resources** | Comprehensive Arabic NLP resource list | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white)](https://github.com/NNLP-IL/Arabic-Resources) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 🏭 Companies & Startups

> Companies and organizations building Arabic AI/NLP products and services.

### 🇦🇪 United Arab Emirates

| Company | Focus | Notable Products | Link |
|:---|:---|:---|:---:|
| **G42** | AI holding company, Arabic LLMs | Jais LLM, enterprise AI solutions | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://www.g42.ai/) |
| **Inception AI** | Arabic-centric foundation models | Jais model family (with Cerebras) | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://www.g42.ai/) |
| **Technology Innovation Institute (TII)** | Open-source LLMs, research | Falcon LLM family | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://www.tii.ae/) |
| **Saal.ai** | Cognitive AI solutions | Arabic NLP, speech, generative AI | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://saal.ai/) |
| **Arabot** | Conversational AI for Arabic | Arabic NLP chatbot engine | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://arabot.io/) |
| **Arabic.AI (Tarjama)** | Arabic-first autonomous AI | Pronoia Arabic LLM, Agentic AI platform | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://www.tarjama.com/) |
| **DXwand** | AI-powered chatbots and analytics | ORXTRA platform, Arabic/English NLU | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://dxwand.com/) |

### 🇸🇦 Saudi Arabia

| Company | Focus | Notable Products | Link |
|:---|:---|:---|:---:|
| **SDAIA** | Sovereign AI, national data authority | ALLaM model, HUMAIN initiative | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://sdaia.gov.sa/) |
| **Future Look ITC (FLITC)** | Arabic-native AI solutions, venture studio | LABEAH, Smart Hire, Rayee Media, Nabadat | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://flitc.ai/) |
| **Wittify.ai** | Conversational AI for Arabic | Interactive Arabic AI agents | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://wittify.ai/) |
| **Lucidya** | AI customer experience analytics | Arabic social listening, sentiment analysis | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://www.lucidya.com/) |

### 🇪🇬 Egypt

| Company | Focus | Notable Products | Link |
|:---|:---|:---|:---:|
| **Velents** | Enterprise Arabic AI solutions | Agent.sa (Arabic-speaking AI employee) | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://velents.com/) |

### 🌍 International (with Arabic focus)

| Company | Country | Focus | Notable Products | Link |
|:---|:---|:---|:---|:---:|
| **SILMA AI** | - | Arabic-first LLMs | SILMA LLMs, Arabic Broad Benchmark | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://silma.ai/) |
| **Cohere** | Canada | Multilingual LLMs | Command R Arabic, RAG optimization | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://cohere.com/) |
| **Mistral AI** | France | Multilingual LLMs | Mistral Saba (Arabic-optimized) | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://mistral.ai/) |
| **SambaNova Systems** | USA | Arabic language adaptation | SambaLingo Arabic models | [![Web](https://img.shields.io/badge/-Website-4285F4?logo=googlechrome&logoColor=white)](https://sambanova.ai/) |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 📚 Awesome Lists

| List | Description |
|:---|:---|
| [01walid/awesome-arabic](https://github.com/01walid/awesome-arabic) | Awesome projects, libraries, and resources for Arabic |
| [Curated-Awesome-Lists/awesome-arabic-nlp](https://github.com/Curated-Awesome-Lists/awesome-arabic-nlp) | Comprehensive Arabic NLP resources |
| [MoroccoAI/Arabic-Darija-NLP-Resources](https://github.com/MoroccoAI/Arabic-Darija-NLP-Resources) | Moroccan Arabic dialect NLP resources |
| [NNLP-IL/Arabic-Resources](https://github.com/NNLP-IL/Arabic-Resources) | Comprehensive Arabic NLP resource list |

<div align="right"><a href="#-table-of-contents">⬆ Back to Top</a></div>

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request to add new resources or update existing ones.

1. Fork the repository
2. Add your resource in the appropriate section
3. Ensure links are valid and descriptions are concise
4. Submit a pull request

---

## 📜 License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

---

## 📑 References

[1] The Landscape of Arabic Large Language Models. (2025). *arXiv preprint arXiv:2506.01340*. https://arxiv.org/html/2506.01340v1

[2] Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding. *OSACT*. https://aclanthology.org/2020.osact-1.2/

[3] Inoue, G., et al. (2021). The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models. *WANLP*. https://aclanthology.org/2021.wanlp-1.10/

[4] Abdul-Mageed, M., et al. (2021). ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic. *ACL*. https://aclanthology.org/2021.acl-long.551/

[5] Sengupta, N., et al. (2023). Jais and Jais-chat. *arXiv:2308.16149*. https://arxiv.org/abs/2308.16149

[6] Jarrar, M., et al. (2022). Wojood: Nested Arabic Named Entity Corpus. *arXiv:2205.09651*. https://arxiv.org/abs/2205.09651

[7] Al-Ayyoub, M., et al. (2018). Deep learning for Arabic NLP: A survey. *Journal of Computational Science*, 26. https://www.sciencedirect.com/science/article/pii/S1877750317303757

[8] Alzubaidi, A., et al. (2025). Evaluating Arabic LLMs: Benchmarks, Methods, and Gaps. *arXiv:2510.13430*. https://arxiv.org/abs/2510.13430

[9] Bhatia, G., et al. (2024). Swan and ArabicMTEB. *arXiv:2411.01192*. https://arxiv.org/abs/2411.01192

[10] Nacar, O., et al. (2025). GATE: General Arabic Text Embedding. *arXiv:2505.24581*. https://arxiv.org/abs/2505.24581

[11] Mashaabi, M., et al. (2024). A Survey of LLMs for Arabic. *arXiv:2410.20238*. https://arxiv.org/abs/2410.20238

[12] Hate speech detection in Arabic. (2024). *Frontiers in AI*. https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1345445/full

[13] NADI 2024: Fifth Nuanced Arabic Dialect Identification. (2024). *ArabicNLP*. https://arxiv.org/abs/2407.04910

[14] CATT: Character-based Arabic Tashkeel Transformer. (2024). *arXiv:2407.03236*. https://arxiv.org/abs/2407.03236

[15] ArabianGPT: Native Arabic GPT-based LLM. (2024). *arXiv:2402.15313*. https://arxiv.org/abs/2402.15313

[16] SambaLingo: Teaching LLMs New Languages. (2024). *arXiv:2404.05829*. https://arxiv.org/abs/2404.05829

---

<div align="center">

**If you find this resource helpful, please give it a ⭐**

Made with ❤️ for the Arabic NLP community

</div>
