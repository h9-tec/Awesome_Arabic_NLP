
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
- [Key Datasets](#key-datasets)
- [Essential Tools & Libraries](#essential-tools--libraries)
- [Top Research Papers & Conferences](#top-research-papers--conferences)
- [Awesome Lists](#awesome-lists)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Key Organizations

| Organization | Focus | Key Contributions | Link |
|---|---|---|---|
| **AUB MIND Lab** | Foundational Arabic NLP models | AraBERT, AraGPT2, AraELECTRA | [GitHub](https://github.com/aub-mind) |
| **CAMeL Lab, NYUAD** | Arabic NLP tools and models | CAMeLBERT, camel_tools | [GitHub](https://github.com/CAMeL-Lab) |
| **UBC-NLP** | Dialectal Arabic, multimodal models | MARBERT, NileChat, PEARL, Dallah | [Hugging Face](https://huggingface.co/UBC-NLP) |
| **QCRI** | Arabic LLMs, knowledge editing | Fanar LLMs, AraDiCE | [Hugging Face](https://huggingface.co/QCRI) |
| **SILMA AI** | State-of-the-art Arabic LLMs | SILMA LLMs, Arabic Broad Benchmark | [Website](https://silma.ai/) |
| **MBZUAI** | Multimodal and speech models | AIN, ArTST, ClArTTS | [Hugging Face](https://huggingface.co/MBZUAI) |
| **ARBML** | Democratizing Arabic NLP | masader, klaam, tkseem | [GitHub](https://github.com/ARBML) |
| **NAMAA-Space** | OCR and Egyptian Arabic models | Qari-OCR, EgypTalk-ASR | [Hugging Face](https://huggingface.co/NAMAA-Space) |
| **Omartificial-Intelligence-Space** | Arabic embedding models | GATE, Matryoshka embeddings | [Hugging Face](https://huggingface.co/Omartificial-Intelligence-Space) |
| **TII (Technology Innovation Institute)** | Arabic LLM benchmarks | Open Arabic LLM Leaderboard | [Website](https://www.tii.ae/) |
| **SDAIA (Saudi Data & AI Authority)** | Sovereign Arabic LLM | ALLaM model | [Website](https://sdaia.gov.sa/) |
| **SinaLab, Birzeit University** | Arabic NLP tools and datasets | SinaTools, Wojood NER | [GitHub](https://github.com/SinaLab) |

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

## State-of-the-Art Models

### Large Language Models (LLMs)

| Model | Parameters | Developed By | Key Features | Link |
|---|---|---|---|---|
| **Jais** | 13B, 30B | Inception AI, Cerebras | Arabic-centric, bilingual (Arabic/English), instruction-tuned chat version | [Hugging Face](https://huggingface.co/inceptionai/jais-30b-v3) |
| **SILMA 1.0** | 9B | SILMA AI | Top-ranked Arabic LLM built on Google Gemma | [Hugging Face](https://huggingface.co/silma-ai/SILMA-9B-Instruct-v1.0) |
| **ALLaM** | 7B | SDAIA & IBM | Saudi's Sovereign Model | [Hugging Face](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview) |
| **Fanar-1-9B** | 9B | QCRI | Arabic-English LLM | [Hugging Face](https://huggingface.co/QCRI/Fanar-1-9B-Instruct) |
| **AceGPT** | 7B | FreedomIntelligence | Top performance on Arabic benchmarks | [GitHub](https://github.com/FreedomIntelligence/AceGPT) |
| **Atlas-Chat** | 2B, 9B, 27B | MBZUAI-IFM Paris Lab| LLM family for Moroccan (Darija) dialect | [Hugging Face](https://huggingface.co/collections/MBZUAI-Paris/atlas-chat) |
| **NileChat-3B** | 3B | UBC-NLP | LLM for Egyptian and Moroccan dialects | [Hugging Face](https://huggingface.co/UBC-NLP/NileChat-3B) |
| **Nile-Chat** | 4B, 3x4B-A6B, 12B | MBZUAI-IFM Paris Lab| LLM family for Egyptian Arabic and Latin (Arabizi) scripts | [Hugging Face](https://huggingface.co/collections/MBZUAI-Paris/nile-chat) |
| **AraGPT2** | 1.5B | AUB MIND Lab | GPT-2 for Arabic text generation | [Hugging Face](https://huggingface.co/aubmindlab/aragpt2-mega) |
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
| **CAMeLBERT** | CAMeL Lab, NYUAD | Models for MSA, Dialectal, and Classical Arabic | [GitHub](https://github.com/CAMeL-Lab/CAMeLBERT) |
| **MARBERT** | UBC-NLP | Focused on Dialectal Arabic and MSA | [GitHub](https://github.com/UBC-NLP/marbert) |
| **AraELECTRA** | AUB MIND Lab | ELECTRA for Arabic | [Hugging Face](https://huggingface.co/aubmindlab/araelectra-base-discriminator) |

### Embedding Models

| Model | Developer | Key Features | Link |
|---|---|---|---|
| **BGE-M3** | BAAI | Multilingual (100+ langs including Arabic), Multi-Functionality (Dense, Sparse, Multi-vector), 8192 token context | [Hugging Face](https://huggingface.co/BAAI/bge-m3) |
| **GATE-AraBert-v1** | Omartificial-Intelligence-Space | General Arabic Text Embedding, SOTA on MTEB Arabic STS | [Hugging Face](https://huggingface.co/Omartificial-Intelligence-Space/GATE-AraBert-v1) |
| **Arabic-Triplet-Matryoshka-V2** | Omartificial-Intelligence-Space | Matryoshka representation for efficient embedding | [Hugging Face](https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2) |
| **Swan** | UBC-NLP | Dialect-aware and cross-lingual (Small & Large versions) | [Paper](https://arxiv.org/abs/2411.01192) |
| **asafaya/bert-base-arabic** | asafaya | BERT-based Arabic embeddings | [Hugging Face](https://huggingface.co/asafaya/bert-base-arabic) |


### Task-Specific Models

This section provides a non-exhaustive list of models for various tasks. For a complete list, please refer to the Hugging Face Hub.

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

### Speech Datasets

| Dataset | Description | Link |
|---|---|---|
| **ClArTTS** | Classical Arabic TTS dataset | [Hugging Face](https://huggingface.co/datasets/MBZUAI/ClArTTS) |
| **MADIS5** | Spoken Arabic dialects dataset | [Hugging Face](https://huggingface.co/datasets/badrex/MADIS5-spoken-arabic-dialects) |
| **SADA22** | MSA and Khaliji speech datasets | [Hugging Face](https://huggingface.co/datasets/badrex/arabic-speech-SADA22-MSA) |
| **Arabic Spontaneous Dialogue Dataset** | 300+ hours of real conversations | [Defined.ai](https://defined.ai/datasets/arabic-spontaneous-dialogue) |

### Vision & Multimodal Datasets

| Dataset | Description | Link |
|---|---|---|
| **PEARL** | Multimodal Culturally-Aware Arabic Instruction Dataset | [Hugging Face](https://huggingface.co/datasets/UBC-NLP/PEARL) |
| **Arabic-Image-Captioning_100M** | 100 million Arabic image captions | [Hugging Face](https://huggingface.co/datasets/Misraj/Arabic-Image-Captioning_100M) |
| **Calliar** | Online Arabic calligraphy dataset (2500 samples) | [GitHub](https://github.com/ARBML/Calliar) |

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

### Specialized Libraries

| Library | Task | Link |
|---|---|---|
| **klaam** | Speech Recognition, Classification, TTS | [GitHub](https://github.com/ARBML/klaam) |
| **tkseem** | Arabic Tokenization | [GitHub](https://github.com/ARBML/tkseem) |
| **arabic-stop-words** | Largest list of Arabic stop words | [GitHub](https://github.com/mohataher/arabic-stop-words) |
| **qawafi** | Arabic poetry analysis | [GitHub](https://github.com/ARBML/qawafi) |

## Top Research Papers & Conferences

### Conferences

| Conference | Year | Link |
|---|---|---|
| **ArabicNLP 2025** | 2025 | [Website](https://arabicnlp2025.sigarab.org/) |
| **ArabicNLP 2024** | 2024 | [ACL Anthology](https://aclanthology.org/events/arabicnlp-2024/) |
| **ArabicNLP 2023** | 2023 | [ACL Anthology](https://aclanthology.org/venues/arabicnlp/) |

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

## Awesome Lists

- [01walid/awesome-arabic](https://github.com/01walid/awesome-arabic) - A curated list of awesome projects, libraries, and resources for the Arabic language.
- [Curated-Awesome-Lists/awesome-arabic-nlp](https://github.com/Curated-Awesome-Lists/awesome-arabic-nlp) - A comprehensive list of Arabic NLP resources.

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