# NeMo Curator: Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [Purpose and Core Value](#purpose-and-core-value)
3. [Key Features](#key-features)
4. [Architecture and Components](#architecture-and-components)
5. [Modality-Specific Capabilities](#modality-specific-capabilities)
6. [Data Curation Fundamentals](#data-curation-fundamentals)
   - [What is Data Curation?](#what-is-data-curation)
   - [Why Data Curation is Critical](#why-data-curation-is-critical)
   - [Components of Data Curation](#components-of-data-curation)
   - [Quality Filtering](#quality-filtering-1)
   - [Deduplication](#deduplication-1)
   - [PII Identification and Removal](#pii-identification-and-removal-1)
   - [Language Identification and Separation](#language-identification-and-separation-1)
   - [Downstream Task Decontamination](#downstream-task-decontamination-1)
   - [Synthetic Data Generation](#synthetic-data-generation-1)
7. [Installation](#installation)
8. [Code Examples](#code-examples)
9. [Performance Characteristics](#performance-characteristics)
10. [Use Cases](#use-cases)
11. [Resources](#resources)

---

## Overview

**NeMo Curator** is an open-source, GPU-accelerated data curation platform developed by NVIDIA for preparing large-scale datasets across multiple modalities. It serves as a scalable toolkit designed to process and prepare datasets for training better AI models, faster.

### What is NeMo Curator?

NeMo Curator is a comprehensive data processing framework that enables data scientists and ML engineers to:
- Curate massive datasets across **text, image, video, and audio** modalities
- Leverage **GPU acceleration** for significantly faster processing
- Scale from individual machines to **distributed multi-node GPU clusters**
- Build reproducible, production-grade data curation pipelines

### Current Version
- **Latest Release**: 1.0.0 (Released October 1, 2025)
- **Documentation Version**: 25.09
- **Python Requirements**: Python >=3.10, <3.13

---

## Purpose and Core Value

NeMo Curator addresses the critical challenge of **data quality** in AI model training. The platform is purpose-built for:

### Primary Use Cases
1. **Foundation Language Model Pretraining** - Curating trillion-token datasets for large language models
2. **Text-to-Image Model Training** - Processing large-scale image-text pairs for multimodal models
3. **Domain-Adaptive Pretraining (DAPT)** - Preparing domain-specific datasets
4. **Supervised Fine-Tuning (SFT)** - Curating high-quality instruction datasets
5. **Parameter-Efficient Fine-Tuning (PEFT)** - Optimizing smaller, focused datasets
6. **World Foundation Models** - Processing video data for spatiotemporal understanding
7. **Speech Model Training** - Curating audio datasets for ASR and speech synthesis

### Target Audience
- **Data Scientists** - Building and experimenting with curation pipelines
- **Machine Learning Engineers** - Deploying production-scale data workflows
- **Cluster Administrators** - Managing distributed GPU infrastructure
- **DevOps Professionals** - Orchestrating automated curation systems

---

## Key Features

### 1. Multi-Modal Support
NeMo Curator provides comprehensive support for four primary data modalities:
- **Text**: Language model datasets (Common Crawl, Wikipedia, ArXiv, custom sources)
- **Image**: Vision-language datasets and image generation training data
- **Video**: Spatiotemporal datasets for world models and video generation
- **Audio**: Speech recognition and synthesis datasets

### 2. GPU Acceleration
- Built on **NVIDIA RAPIDS** libraries (cuDF, cuML, cuGraph)
- Leverages GPU computing for massive speedups over CPU-based solutions
- **16× faster** fuzzy deduplication on large datasets
- **40% lower** total cost of ownership vs. CPU alternatives

### 3. Scalable Architecture
- **Single Machine**: Process datasets on individual GPUs
- **Multi-Node Clusters**: Near-linear scaling across GPU nodes
- **Distributed Computing**: Built on Ray framework for horizontal scaling
- **Flexible Deployment**: Containers available via NVIDIA NGC

### 4. Comprehensive Processing Capabilities

#### Quality Assessment
- 30+ heuristic filters for text quality
- GPU-accelerated classifiers
- Aesthetic quality scoring for images
- Motion and quality filtering for video
- Word Error Rate (WER) calculation for audio

#### Deduplication
- **Exact Deduplication**: Hash-based duplicate detection
- **Fuzzy Deduplication**: MinHash LSH for near-duplicate detection
- **Semantic Deduplication**: Embedding-based similarity clustering

#### Content Filtering
- Language identification and filtering
- NSFW content detection
- Domain-specific classification
- Custom filter development

---

## Architecture and Components

### System Architecture

NeMo Curator follows a **modular, pipeline-based architecture** with several key layers:

```
┌─────────────────────────────────────────────────────────┐
│              User Interface / API Layer                 │
├─────────────────────────────────────────────────────────┤
│           Pipeline Orchestration Layer                  │
│         (Compose stages into workflows)                 │
├─────────────────────────────────────────────────────────┤
│              Processing Stages Layer                    │
│  ┌──────────┬──────────┬──────────┬──────────┐        │
│  │  Text    │  Image   │  Video   │  Audio   │        │
│  │  Stages  │  Stages  │  Stages  │  Stages  │        │
│  └──────────┴──────────┴──────────┴──────────┘        │
├─────────────────────────────────────────────────────────┤
│            Execution Backends Layer                     │
│         (Ray, Xenna, distributed compute)              │
├─────────────────────────────────────────────────────────┤
│          Core Data Structures Layer                     │
│  (Documents, Images, Videos, AudioBatches)             │
├─────────────────────────────────────────────────────────┤
│         Infrastructure Layer                            │
│  (RAPIDS: cuDF, cuML, cuGraph + Dask)                  │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Execution Backends**
The execution layer provides distributed computing capabilities:
- **Ray-based Backends**: Scalable distributed execution framework
- **Xenna Executor**: Experimental execution framework
- **Adapters**: Integration points for custom execution environments

**Key Features:**
- Horizontal scaling across multiple nodes
- Resource management and scheduling
- Fault tolerance and checkpointing
- Dynamic resource allocation

#### 2. **Pipeline Orchestration**
Tools for building end-to-end data curation workflows:
- **Pipeline Builder**: Compose processing stages into workflows
- **Stage Management**: Control execution order and dependencies
- **Data Flow**: Pass data objects between stages
- **Configuration**: Parameterize pipelines for different datasets

**Usage Pattern:**
```
Data Input → Stage 1 → Stage 2 → ... → Stage N → Output
```

#### 3. **Processing Stages**
Modular components organized by modality and function:

**Text Stages:**
- Classifiers (quality, domain, language)
- Deduplication (exact, fuzzy, semantic)
- Downloads (Common Crawl, Wikipedia, ArXiv)
- Filters (heuristics, quality metrics)
- Formatters (standardization, tokenization)

**Image Stages:**
- Embedding generation (CLIP, custom models)
- Classification (aesthetic, NSFW, quality)
- Deduplication (perceptual hashing, semantic)
- Filtering (resolution, aspect ratio, quality)
- I/O operations (WebDataset format)

**Video Stages:**
- Scene detection (TransNetV2)
- Clipping and segmentation
- Caption generation
- Embedding extraction (InternVideo2, Cosmos-Embed1)
- Transcoding (GPU-accelerated H.264)
- Frame extraction and sampling

**Audio Stages:**
- ASR inference (NeMo Framework models)
- Quality metrics (WER, duration, SNR)
- Transcription processing
- I/O operations

#### 4. **Core Data Structures**
Standardized objects passed between processing stages:

**Documents (Text)**
- Batch processing of text documents
- Metadata tracking
- Efficient serialization

**Images**
- Image-text pairs
- Metadata and annotations
- WebDataset compatibility

**Videos**
- Video clips and metadata
- Frame sequences
- Caption associations

**AudioBatches**
- Audio samples and transcriptions
- Quality metrics
- Temporal information

**FileGroups**
- Collections of related files
- Distributed file management
- Partition handling

#### 5. **Utility Functions**
Helper functions supporting pipeline operations:
- File operations and I/O
- Performance optimization utilities
- Logging and monitoring
- Distributed computing helpers
- Configuration management

---

## Modality-Specific Capabilities

### Text Data Curation

Text curation in NeMo Curator is designed for processing massive language model datasets efficiently.

#### Key Capabilities

**1. Data Loading and Processing**
- Support for Common Crawl, Wikipedia, ArXiv datasets
- Custom dataset integration
- Distributed file reading
- Format standardization

**2. Quality Filtering**
- **30+ Heuristic Filters** including:
  - Document length (word count, character count)
  - Punctuation ratio
  - Symbol-to-word ratio
  - Alpha ratio (alphabetic characters)
  - Stop word coverage
  - Line length statistics
  - Repetition detection
  - Boilerplate removal

**3. Classification**
- **GPU-accelerated classifiers** for:
  - Domain classification
  - Quality scoring
  - Topic categorization
  - Custom classification tasks

**4. Language Identification**
- Multilingual language detection
- Language-specific filtering
- Character encoding normalization

**5. Deduplication**
Three complementary approaches:

**Exact Deduplication:**
- Hash-based duplicate detection
- Fastest method for identical documents
- Low memory footprint

**Fuzzy Deduplication (MinHash LSH):**
- Near-duplicate detection
- Configurable similarity thresholds
- Scalable to trillion-token datasets
- **16× faster** than CPU alternatives on RedPajama v2

**Semantic Deduplication:**
- Embedding-based similarity
- Captures semantic duplicates
- Higher precision than fuzzy methods

#### Text Processing Pipeline Example

```
Raw Text Corpus
     ↓
Data Loading (Distributed)
     ↓
Format Standardization
     ↓
Language Identification
     ↓
Quality Filtering (Heuristics)
     ↓
Quality Classification (GPU)
     ↓
Exact Deduplication
     ↓
Fuzzy Deduplication (MinHash)
     ↓
Semantic Deduplication
     ↓
Final Curated Dataset
```

---

### Image Data Curation

Image curation focuses on preparing large-scale image-text datasets for vision-language models and generative AI.

#### Key Capabilities

**1. Embedding Generation**
- **CLIP Embeddings**: Semantic representation of images
- Custom model support
- Batch processing on GPU
- Efficient storage and indexing

**2. Classification and Filtering**
- **Aesthetic Quality Scoring**: Identify visually appealing images
- **NSFW Detection**: Filter inappropriate content
- **Resolution Filtering**: Quality threshold enforcement
- **Aspect Ratio Filtering**: Format consistency

**3. Deduplication**
- Perceptual hashing for near-duplicates
- Embedding-based semantic deduplication
- Cluster analysis for similar images

**4. WebDataset Format**
- Industry-standard format for large-scale datasets
- Efficient streaming and loading
- Distributed training compatibility

#### Image Processing Pipeline Example

```
Raw Image-Text Pairs
     ↓
Image Loading (Distributed)
     ↓
Resolution Filtering
     ↓
CLIP Embedding Generation (GPU)
     ↓
Aesthetic Scoring
     ↓
NSFW Detection
     ↓
Semantic Deduplication
     ↓
WebDataset Export
     ↓
Curated Image Dataset
```

---

### Video Data Curation

Video curation enables processing of massive video datasets for world foundation models and video generation systems.

#### Key Capabilities

**1. Video Processing**
- **Scene Detection**: TransNetV2-based scene segmentation
- **Clip Extraction**: Intelligent video segmentation
- **Frame Sampling**: Extract representative frames
- **GPU-accelerated Transcoding**: H.264 encoding

**2. Quality Assessment**
- **Motion Filtering**: Identify static vs. dynamic content
- **Aesthetic Assessment**: Visual quality scoring
- **Resolution and bitrate analysis**

**3. Embedding Generation**
- **InternVideo2**: Spatiotemporal feature extraction
- **Cosmos-Embed1**: Multimodal video embeddings
- Efficient batch processing

**4. Deduplication**
- **k-means Clustering**: Group similar video clips
- **Embedding-based Similarity**: Semantic duplicate detection
- Scene-level deduplication

**5. Caption Generation**
- Automated video captioning
- Caption quality assessment
- Integration with text pipelines

#### Video Processing Pipeline Example

```
Raw Video Collection
     ↓
Video Loading (Distributed)
     ↓
Scene Detection (TransNetV2)
     ↓
Clip Extraction
     ↓
Frame Extraction
     ↓
Motion Filtering
     ↓
Embedding Generation (GPU)
     ↓
Aesthetic Assessment
     ↓
Deduplication (k-means)
     ↓
Caption Generation
     ↓
GPU Transcoding (H.264)
     ↓
Curated Video Dataset
```

---

### Audio Data Curation

Audio curation focuses on speech dataset preparation for ASR and speech synthesis models.

#### Key Capabilities

**1. Automatic Speech Recognition**
- **NeMo Framework ASR Models**: State-of-the-art transcription
- Batch inference on GPU
- Multi-language support
- Speaker diarization

**2. Quality Assessment**
- **Word Error Rate (WER)**: Transcription quality metric
- Duration filtering
- Signal-to-noise ratio analysis
- Silence detection

**3. Text Integration**
- Integration with text curation pipelines
- Transcription cleaning and formatting
- Language identification
- Quality filtering of transcriptions

**4. Audio Processing**
- Format standardization
- Resampling and normalization
- Batch processing optimization

#### Audio Processing Pipeline Example

```
Raw Audio Files
     ↓
Audio Loading (Distributed)
     ↓
Format Standardization
     ↓
Duration Filtering
     ↓
ASR Transcription (GPU)
     ↓
WER Calculation
     ↓
Quality Filtering
     ↓
Text Curation (Integration)
     ↓
Language Identification
     ↓
Curated Speech Dataset
```

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- CUDA 12 (for GPU acceleration)
- NVIDIA GPU (recommended for optimal performance)

### Basic Installation

```bash
# Install base package
pip install nemo-curator
```

### Installation with Modality-Specific Support

NeMo Curator provides optional installation profiles for different modalities and hardware:

```bash
# Text curation with CUDA 12 support
pip install "nemo-curator[text-cuda12]"

# Image curation with CUDA 12
pip install "nemo-curator[image-cuda12]"

# Video processing with CUDA 12
pip install "nemo-curator[video-cuda12]"

# Audio curation with CUDA 12
pip install "nemo-curator[audio-cuda12]"

# Complete installation (all modalities)
pip install "nemo-curator[all]"

# CPU-only installations (slower performance)
pip install "nemo-curator[text-cpu]"
pip install "nemo-curator[image-cpu]"
pip install "nemo-curator[video-cpu]"
pip install "nemo-curator[audio-cpu]"
```

### Using uv Package Manager

```bash
# Recommended for faster installation
uv pip install "nemo-curator[text_cuda12]"
```

### Docker Installation

NeMo Curator containers are available through NVIDIA NGC:

```bash
# Pull the latest container
docker pull nvcr.io/nvidia/nemo:latest

# Run the container
docker run --gpus all -it --rm nvcr.io/nvidia/nemo:latest
```

### Verification

After installation, verify the setup:

```bash
# Run the quickstart tutorial
python -c "import nemo_curator; print(nemo_curator.__version__)"
```

---

## Code Examples

### Example 1: Basic Text Curation Pipeline

```python
from nemo_curator import ScoreFilter, Modify
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import WordCountFilter, NonAlphabeticFilter
from nemo_curator.modifiers import LowercaseModifier

# Load your dataset
dataset = DocumentDataset.read_json("input_data.jsonl", backend="pandas")

# Apply word count filtering
word_count_filter = ScoreFilter(
    WordCountFilter(min_words=50),
    score_field="word_count",
    score_type=int
)
filtered_dataset = word_count_filter(dataset)

# Apply non-alphabetic character filtering
alpha_filter = ScoreFilter(
    NonAlphabeticFilter(max_non_alpha_ratio=0.3),
    score_field="alpha_ratio",
    score_type=float
)
filtered_dataset = alpha_filter(filtered_dataset)

# Apply text modification (lowercase)
modifier = Modify(LowercaseModifier())
modified_dataset = modifier(filtered_dataset)

# Write the results
modified_dataset.to_json("output_data.jsonl", write_to_filename=True)
```

### Example 2: Text Deduplication Pipeline

```python
from nemo_curator import ExactDeduplication
from nemo_curator.datasets import DocumentDataset

# Load dataset
dataset = DocumentDataset.read_json("large_corpus.jsonl", backend="dask")

# Configure exact deduplication
exact_dedup = ExactDeduplication(
    id_field="id",
    text_field="text",
    hash_method="md5"
)

# Run deduplication
deduplicated_dataset = exact_dedup(dataset)

# Save results
deduplicated_dataset.to_json("deduplicated_corpus.jsonl")

print(f"Original documents: {len(dataset)}")
print(f"After deduplication: {len(deduplicated_dataset)}")
print(f"Removed duplicates: {len(dataset) - len(deduplicated_dataset)}")
```

### Example 3: Fuzzy Deduplication with MinHash

```python
from nemo_curator import FuzzyDeduplication
from nemo_curator.datasets import DocumentDataset
import dask_cudf

# Load large dataset with Dask
dataset = DocumentDataset.read_json(
    "trillion_token_corpus/*.jsonl",
    backend="cudf"  # Use GPU-accelerated cuDF
)

# Configure fuzzy deduplication
fuzzy_dedup = FuzzyDeduplication(
    id_field="id",
    text_field="text",
    num_hashes=128,  # Number of MinHash signatures
    num_bands=16,    # LSH bands for similarity grouping
    jaccard_threshold=0.8  # Similarity threshold
)

# Run GPU-accelerated fuzzy deduplication
deduplicated_dataset = fuzzy_dedup(dataset)

# Save results
deduplicated_dataset.to_json("fuzzy_deduplicated_corpus.jsonl")
```

### Example 4: Language Identification and Filtering

```python
from nemo_curator import ScoreFilter
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import LanguageFilter

# Load multilingual dataset
dataset = DocumentDataset.read_json("multilingual_data.jsonl")

# Configure language filter (keep only English)
lang_filter = ScoreFilter(
    LanguageFilter(language="en"),
    score_field="language",
    score_type=str
)

# Apply filter
english_only = lang_filter(dataset)

# Save filtered dataset
english_only.to_json("english_corpus.jsonl")
```

### Example 5: GPU-Accelerated Quality Classification

```python
from nemo_curator import ClassifyDocuments
from nemo_curator.datasets import DocumentDataset
from nemo_curator.classifiers import QualityClassifier

# Load dataset
dataset = DocumentDataset.read_json("raw_corpus.jsonl", backend="cudf")

# Configure quality classifier
quality_classifier = ClassifyDocuments(
    QualityClassifier(
        model_name="roberta-base-quality",
        batch_size=32,
        device="cuda"  # Use GPU
    ),
    score_field="quality_score"
)

# Run classification
classified_dataset = quality_classifier(dataset)

# Filter based on quality score
high_quality = classified_dataset.df[
    classified_dataset.df["quality_score"] > 0.7
]

# Save high-quality subset
high_quality.to_json("high_quality_corpus.jsonl")
```

### Example 6: Image Curation with CLIP Embeddings

```python
from nemo_curator.image import GenerateEmbeddings, AestheticFilter
from nemo_curator.datasets import ImageDataset

# Load image-text dataset
image_dataset = ImageDataset.read_webdataset("images/*.tar")

# Generate CLIP embeddings
embedding_generator = GenerateEmbeddings(
    model_name="openai/clip-vit-large-patch14",
    batch_size=256,
    device="cuda"
)
embedded_dataset = embedding_generator(image_dataset)

# Apply aesthetic filtering
aesthetic_filter = AestheticFilter(
    min_score=0.5,  # Keep aesthetically pleasing images
    score_field="aesthetic_score"
)
filtered_dataset = aesthetic_filter(embedded_dataset)

# Export to WebDataset format
filtered_dataset.to_webdataset("curated_images/")
```

### Example 7: Video Scene Detection and Clipping

```python
from nemo_curator.video import SceneDetection, ClipExtraction
from nemo_curator.datasets import VideoDataset

# Load video dataset
video_dataset = VideoDataset.read_video_list("video_urls.txt")

# Detect scenes using TransNetV2
scene_detector = SceneDetection(
    model="transnetv2",
    threshold=0.5,
    device="cuda"
)
scenes_dataset = scene_detector(video_dataset)

# Extract clips based on scenes
clip_extractor = ClipExtraction(
    min_duration=2.0,  # Minimum 2 seconds
    max_duration=10.0,  # Maximum 10 seconds
    output_format="mp4"
)
clips_dataset = clip_extractor(scenes_dataset)

# Save clip metadata
clips_dataset.to_json("video_clips_metadata.jsonl")
```

### Example 8: Audio Transcription with ASR

```python
from nemo_curator.audio import ASRInference, WERFilter
from nemo_curator.datasets import AudioDataset

# Load audio dataset
audio_dataset = AudioDataset.read_audio_files("audio/*.wav")

# Transcribe using NeMo ASR
asr = ASRInference(
    model_name="stt_en_conformer_ctc_large",
    batch_size=16,
    device="cuda"
)
transcribed_dataset = asr(audio_dataset)

# Filter by Word Error Rate (if reference transcripts available)
wer_filter = WERFilter(
    max_wer=0.15,  # Keep WER < 15%
    reference_field="reference_text",
    hypothesis_field="transcription"
)
high_quality_audio = wer_filter(transcribed_dataset)

# Save results
high_quality_audio.to_json("high_quality_transcriptions.jsonl")
```

### Example 9: Complete Multi-Stage Pipeline

```python
from nemo_curator import Pipeline
from nemo_curator import ScoreFilter, ExactDeduplication, Modify
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    WordCountFilter,
    NonAlphabeticFilter,
    RepeatedLinesFilter
)
from nemo_curator.modifiers import UnicodeNormalizer

# Load dataset
dataset = DocumentDataset.read_json(
    "raw_corpus/*.jsonl",
    backend="cudf"  # GPU acceleration
)

# Create pipeline
pipeline = Pipeline([
    # Stage 1: Unicode normalization
    Modify(UnicodeNormalizer()),

    # Stage 2: Word count filter
    ScoreFilter(
        WordCountFilter(min_words=50, max_words=100000),
        score_field="word_count"
    ),

    # Stage 3: Non-alphabetic character filter
    ScoreFilter(
        NonAlphabeticFilter(max_non_alpha_ratio=0.3),
        score_field="alpha_ratio"
    ),

    # Stage 4: Repeated lines filter
    ScoreFilter(
        RepeatedLinesFilter(max_repeated_line_fraction=0.3),
        score_field="repeated_lines"
    ),

    # Stage 5: Exact deduplication
    ExactDeduplication(
        id_field="id",
        text_field="text"
    )
])

# Run the complete pipeline
curated_dataset = pipeline(dataset)

# Save final output
curated_dataset.to_json("fully_curated_corpus.jsonl")

# Print statistics
print(f"Original size: {len(dataset)}")
print(f"Final size: {len(curated_dataset)}")
print(f"Reduction: {(1 - len(curated_dataset)/len(dataset)) * 100:.2f}%")
```

### Example 10: Distributed Processing with Ray

```python
from nemo_curator import DistributedDataset
from nemo_curator.backends import RayBackend
from nemo_curator import ScoreFilter
from nemo_curator.filters import WordCountFilter
import ray

# Initialize Ray cluster
ray.init(
    num_cpus=64,
    num_gpus=8,
    object_store_memory=100 * 1024 ** 3  # 100 GB
)

# Configure Ray backend
backend = RayBackend(
    num_workers=8,
    gpus_per_worker=1
)

# Load dataset with distributed backend
dataset = DistributedDataset.read_json(
    "massive_corpus/**/*.jsonl",
    backend=backend
)

# Apply filtering with distributed execution
filter_stage = ScoreFilter(
    WordCountFilter(min_words=100),
    score_field="word_count"
)
filtered_dataset = filter_stage(dataset)

# Write output in parallel
filtered_dataset.to_json(
    "filtered_corpus/",
    partition_size="1GB"  # Write 1GB partitions
)

# Shutdown Ray
ray.shutdown()
```

---

## Performance Characteristics

### Benchmark Results

NeMo Curator demonstrates exceptional performance on large-scale datasets:

#### Fuzzy Deduplication Performance
**Dataset**: RedPajama v2 subset (1.78 trillion tokens, 8 TB)
- **NeMo Curator (GPU)**: ~6 hours on 8× H100 80GB GPUs
- **CPU Baseline**: ~96 hours on equivalent CPU cluster
- **Speedup**: **16× faster** with GPU acceleration

#### Cost Efficiency
- **40% lower** total cost of ownership compared to CPU-based solutions
- Reduced infrastructure requirements through GPU acceleration
- Energy efficiency gains from shorter processing times

### Scaling Characteristics

#### Near-Linear Scaling
NeMo Curator exhibits near-linear scaling across multiple GPU nodes:

| GPUs | Processing Time | Scaling Efficiency |
|------|----------------|-------------------|
| 1× H100 | 48 hours | 100% (baseline) |
| 2× H100 | 25 hours | 96% |
| 4× H100 | 13 hours | 92% |
| 8× H100 | 6.5 hours | 89% |

#### Throughput Metrics

**Text Processing:**
- ~100M documents/hour (filtering)
- ~50M documents/hour (fuzzy deduplication)
- ~20M documents/hour (semantic deduplication)

**Image Processing:**
- ~1M images/hour (CLIP embedding generation)
- ~5M images/hour (aesthetic filtering)
- ~2M images/hour (semantic deduplication)

**Video Processing:**
- ~100 hours of video/hour (scene detection)
- ~50 hours of video/hour (embedding generation)
- ~200 hours of video/hour (transcoding H.264)

**Audio Processing:**
- ~1000 hours of audio/hour (ASR transcription)
- ~5000 hours of audio/hour (quality filtering)

### Memory Efficiency

NeMo Curator optimizes memory usage through:
- **Streaming Processing**: Process data in chunks to avoid memory overflow
- **Distributed Memory**: Leverage cluster memory across nodes
- **GPU Memory Management**: Efficient VRAM utilization
- **Partition Control**: Configurable partition sizes for different workloads

### Quality Impact

Ablation studies on language model pretraining demonstrate measurable improvements:

| Curation Stage | Downstream Perplexity | Quality Gain |
|----------------|----------------------|--------------|
| Raw Data | 15.2 | Baseline |
| + Cleaning | 14.1 | 7.2% |
| + Deduplication | 13.3 | 12.5% |
| + Quality Filtering | 12.4 | 18.4% |

---

## Use Cases

### 1. Large Language Model Pretraining

**Scenario**: Preparing a trillion-token dataset for foundation model training

**Pipeline:**
```
Common Crawl Download
     ↓
Text Extraction & Cleaning
     ↓
Language Identification
     ↓
Quality Heuristic Filtering
     ↓
GPU-Accelerated Quality Classification
     ↓
Exact Deduplication
     ↓
Fuzzy Deduplication (MinHash)
     ↓
Semantic Deduplication
     ↓
Final Dataset (300B high-quality tokens)
```

**Benefits:**
- Process trillion-token datasets in days, not weeks
- Improve model performance through quality filtering
- Reduce training costs by eliminating redundant data

### 2. Vision-Language Model Training

**Scenario**: Curating a 1 billion image-text pair dataset

**Pipeline:**
```
Web-Scraped Image-Text Pairs
     ↓
Resolution & Format Filtering
     ↓
CLIP Embedding Generation (GPU)
     ↓
Aesthetic Quality Scoring
     ↓
NSFW Content Filtering
     ↓
Image-Text Alignment Scoring
     ↓
Semantic Deduplication
     ↓
WebDataset Export
     ↓
Final Dataset (200M high-quality pairs)
```

**Benefits:**
- Filter low-quality and inappropriate content
- Improve vision-language alignment
- Optimize dataset size for efficient training

### 3. Domain-Adaptive Pretraining

**Scenario**: Creating a specialized medical domain dataset

**Pipeline:**
```
Medical Literature (PubMed, ArXiv)
     ↓
Domain Classification (Medical vs. Non-Medical)
     ↓
Entity Recognition Quality Check
     ↓
Citation and Reference Validation
     ↓
Terminology Consistency Check
     ↓
Deduplication
     ↓
Medical Domain Dataset
```

**Benefits:**
- High-precision domain filtering
- Maintain domain-specific quality
- Create focused datasets for specialized models

### 4. Instruction Fine-Tuning Dataset Creation

**Scenario**: Curating high-quality instruction-following examples

**Pipeline:**
```
Raw Instruction-Response Pairs
     ↓
Format Validation
     ↓
Instruction Clarity Scoring
     ↓
Response Quality Assessment
     ↓
Diversity Analysis
     ↓
Deduplication (Exact & Semantic)
     ↓
Balanced Sampling
     ↓
Fine-Tuning Dataset
```

**Benefits:**
- Ensure instruction quality and diversity
- Remove low-quality or harmful examples
- Create balanced, representative datasets

### 5. Video World Model Training

**Scenario**: Processing 1 million hours of video for spatiotemporal learning

**Pipeline:**
```
Raw Video Collection
     ↓
Scene Detection & Segmentation
     ↓
Motion & Quality Filtering
     ↓
Frame Extraction & Sampling
     ↓
Embedding Generation (InternVideo2)
     ↓
Semantic Clustering & Deduplication
     ↓
Caption Generation
     ↓
Curated Video Clips
```

**Benefits:**
- Extract meaningful video segments
- Filter static or low-quality content
- Create diverse, high-quality video datasets

### 6. Multilingual Dataset Creation

**Scenario**: Building balanced multilingual training data

**Pipeline:**
```
Multilingual Web Scrape
     ↓
Language Identification
     ↓
Per-Language Quality Filtering
     ↓
Translation Quality Assessment
     ↓
Cross-Lingual Deduplication
     ↓
Language Balance Sampling
     ↓
Multilingual Dataset
```

**Benefits:**
- Accurate language identification
- Language-specific quality standards
- Balanced representation across languages

---
## Resources

### Official Documentation
- **Main Documentation**: https://docs.nvidia.com/nemo/curator/latest/
- **API Reference**: https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html
- **Concepts Guide**: https://docs.nvidia.com/nemo/curator/latest/about/concepts/index.html
- **Installation Guide**: https://docs.nvidia.com/nemo/curator/latest/admin/installation.html
- **Infrastructure Guide**: https://docs.nvidia.com/nemo/curator/latest/reference/infrastructure/index.html

### Code Repository
- **GitHub**: https://github.com/NVIDIA/NeMo-Curator
- **Issues**: https://github.com/NVIDIA/NeMo-Curator/issues
- **Discussions**: https://github.com/NVIDIA/NeMo-Curator/discussions

### Tutorials
- **Text Curation**: `/tutorials/text/` in repository
- **Image Curation**: `/tutorials/image/` in repository
- **Video Processing**: `/tutorials/video/` in repository
- **Audio Workflows**: `/tutorials/audio/` in repository
- **Quickstart**: `tutorials/quickstart.py`

### Container Images
- **NVIDIA NGC Catalog**: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo
- Pull command: `docker pull nvcr.io/nvidia/nemo:latest`

### Package Information
- **PyPI**: https://pypi.org/project/nemo-curator/
- **Latest Version**: 1.0.0
- **License**: Open Source

### Community Support
- **GitHub Discussions**: Community Q&A and feature discussions
- **GitHub Issues**: Bug reports and feature requests
- **NVIDIA Developer Forums**: Official NVIDIA support

### Related Tools
- **NeMo Framework**: https://github.com/NVIDIA/NeMo - End-to-end framework for model training
- **NVIDIA RAPIDS**: https://rapids.ai/ - GPU-accelerated data science
- **Ray**: https://www.ray.io/ - Distributed computing framework

### Research Papers
NeMo Curator implements techniques from cutting-edge research in data curation, deduplication, and quality filtering. Refer to the documentation for specific paper citations.

### Migration Guides
- **Version 25.09 Migration**: Available in official documentation
- **API Changes**: Documented in release notes

---

## Conclusion

NeMo Curator represents a comprehensive, production-ready solution for large-scale data curation across multiple modalities. Its GPU acceleration, distributed computing capabilities, and modular architecture make it an essential tool for:

- **Efficiency**: Process trillion-token datasets 16× faster than CPU alternatives
- **Quality**: Improve model performance through rigorous data curation
- **Scale**: Near-linear scaling from single GPUs to multi-node clusters
- **Flexibility**: Support for text, image, video, and audio modalities
- **Cost**: 40% lower TCO compared to CPU-based solutions

Whether you're preparing datasets for foundation model pretraining, fine-tuning specialized models, or building multimodal AI systems, NeMo Curator provides the tools and performance needed for state-of-the-art data preparation.

---

**Last Updated**: January 2026
**Version**: Based on NeMo Curator 1.0.0 (Release 25.09)

