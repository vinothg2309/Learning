


chapter_9_current_trend.md
---

<!-- TOC start -->
## Table of Contents

- [Resources](#resources)
- [Vision Transformer (ViT)](#vision-transformer-vit)
  - [1. Overview](#1-overview)
  - [2. Architecture \& Visual Explanation](#2-architecture--visual-explanation)
    - [Architecture Diagram](#architecture-diagram)
    - [Patch Embedding Process](#patch-embedding-process)
    - [Performance \& Implementation](#performance--implementation)
  - [3. How Vision Transformer Works](#3-how-vision-transformer-works)
    - [Step-by-Step Process](#step-by-step-process)
    - [Processing Image and Text](#processing-image-and-text)
  - [4. Key Advantages](#4-key-advantages)
  - [5. ViT vs CNNs: Conceptual Comparison](#5-vit-vs-cnns-conceptual-comparison)
    - [Traditional CNNs](#traditional-cnns)
    - [Vision Transformers](#vision-transformers)
    - [Simple Analogy](#simple-analogy)
  - [6. Key Takeaways](#6-key-takeaways)
  - [7. Performance \& Implementation: Deep Dive](#7-performance--implementation-deep-dive)
    - [7.1 Architecture Specifications](#71-architecture-specifications)
      - [ViT Model Variants](#vit-model-variants)
    - [7.2 Mathematical Architecture Breakdown](#72-mathematical-architecture-breakdown)
      - [**Input Processing**](#input-processing)
    - [7.3 Transformer Encoder Architecture](#73-transformer-encoder-architecture)
      - [**Single Transformer Block**](#single-transformer-block)
      - [**Multi-Head Self-Attention Detail**](#multi-head-self-attention-detail)
    - [7.4 Step-by-Step Implementation Example](#74-step-by-step-implementation-example)
      - [**Complete Forward Pass**](#complete-forward-pass)
    - [7.5 Performance Benchmarks](#75-performance-benchmarks)
      - [**ImageNet-1K Results (Top-1 Accuracy)**](#imagenet-1k-results-top-1-accuracy)
    - [7.6 Implementation Considerations](#76-implementation-considerations)
      - [**Memory Requirements**](#memory-requirements)
      - [**Training Configuration**](#training-configuration)
      - [**Pre-training vs Fine-tuning**](#pre-training-vs-fine-tuning)
    - [7.7 Practical Code Example](#77-practical-code-example)
    - [7.8 Key Implementation Insights](#78-key-implementation-insights)
- [Diffusion-based Models](#diffusion-based-models)
  - [1. What Are Diffusion Models?](#1-what-are-diffusion-models)
  - [2. How They Work (Simple Explanation)](#2-how-they-work-simple-explanation)
    - [Training Phase: Learn to Remove Noise](#training-phase-learn-to-remove-noise)
    - [Generation Phase: Create New Images](#generation-phase-create-new-images)
  - [3. Real-World Examples](#3-real-world-examples)
  - [4. Why They're Powerful](#4-why-theyre-powerful)
  - [5. Simple Analogy](#5-simple-analogy)
  - [6. Key Takeaway](#6-key-takeaway)
  - [7. Visual Explanations](#7-visual-explanations)
    - [Auto-Regressive vs Diffusion Models](#auto-regressive-vs-diffusion-models)
    - [Diffusion Process Visualization](#diffusion-process-visualization)
    - [Diffusion Model Architecture \& Training](#diffusion-model-architecture--training)
<!-- TOC end -->

---

# Resources

| **Type** | **Link** |
|----------|----------|
| **Course** | https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9 |
| **Material** | https://cme295.stanford.edu/syllabus/ |


# Vision Transformer (ViT)

## 1. Overview

**Definition:** Vision Transformer (ViT) applies the Transformer architecture (originally designed for text/NLP) to computer vision tasks. Instead of using traditional Convolutional Neural Networks (CNNs), ViT treats images as sequences of patches, similar to how words form sentences.

**Key Insight:** Images can be processed using the same self-attention mechanism that powers language models.

---

## 2. Architecture & Visual Explanation

### Architecture Diagram
![alt text](image.png)

*Overall ViT architecture showing image-to-patches pipeline and transformer processing*

### Patch Embedding Process
![alt text](image-1.png)

*How image patches are converted to embeddings with positional encoding*

### Performance & Implementation
![alt text](image-2.png)


---

## 3. How Vision Transformer Works

### Step-by-Step Process

![alt text](image-8.png)

1. **Image Patching**
   - Divide input image into fixed-size patches (typically 16×16 pixels)
   - Each patch becomes a basic processing unit(Red, Green, Blue)

![alt text](image-7.png)

2. **Patch Embedding**
   - Flatten each 2D patch into a 1D vector
   - Apply linear projection to create patch embeddings


3. **Positional Encoding**
   - Add learnable positional embeddings to each patch
   - Preserves spatial information about patch location in original image

4. **Transformer Processing**
   - Feed patch embeddings through standard Transformer encoder
   - Self-attention mechanism learns relationships between all patches
   - Multiple layers progressively refine representations

5. **Classification Head**
   - Special [CLS] token aggregates information from all patches
   - Final MLP head produces classification output

---

### Processing Image and Text

![alt text](image-9.png)

![alt text](image-10.png)


---

## 4. Key Advantages

| **Advantage** | **Description** |
|---------------|-----------------|
| **Long-range Dependencies** | Captures relationships between distant image regions better than CNNs with limited receptive fields |
| **Scalability** | Performance improves dramatically with larger datasets (unlike CNNs that plateau sooner) |
| **Flexibility** | Same architecture works across diverse vision tasks with minimal modifications |
| **Architectural Simplicity** | Cleaner design compared to complex CNN architectures with residual connections |
| **Transfer Learning** | Pre-trained ViT models transfer exceptionally well to downstream tasks |

---

## 5. ViT vs CNNs: Conceptual Comparison

### Traditional CNNs
- **Approach:** Sliding window with local receptive fields
- **Processing:** Hierarchical feature extraction (edges → textures → objects)
- **Strength:** Strong inductive biases for spatial locality
- **Limitation:** Limited long-range dependency modeling

### Vision Transformers
- **Approach:** Global self-attention across all image patches
- **Processing:** Direct modeling of patch-to-patch relationships
- **Strength:** Captures long-range dependencies from the first layer
- **Limitation:** Requires more data to learn spatial priors

### Simple Analogy
- **CNNs:** Examining a puzzle by looking at one piece and its neighbors at a time
- **ViT:** Looking at all puzzle pieces simultaneously and understanding how they all relate to each other

---

## 6. Key Takeaways

✅ **Paradigm Shift:** ViT demonstrates that Transformers aren't just for NLP—they excel at vision tasks too

✅ **Data-Hungry but Powerful:** Requires large-scale pre-training but achieves state-of-the-art results

✅ **Foundation for Multimodal Models:** ViT's success paved the way for unified image-text models (CLIP, DALL-E, etc.)

✅ **Future Direction:** Represents a move toward more general-purpose, architecture-agnostic deep learning

---

## 7. Performance & Implementation: Deep Dive

### 7.1 Architecture Specifications

#### ViT Model Variants

| **Model** | **Patch Size** | **Layers** | **Hidden Size (D)** | **MLP Size** | **Heads** | **Parameters** |
|-----------|---------------|------------|---------------------|--------------|-----------|----------------|
| ViT-Base  | 16×16         | 12         | 768                 | 3072         | 12        | 86M            |
| ViT-Large | 16×16         | 24         | 1024                | 4096         | 16        | 307M           |
| ViT-Huge  | 14×14         | 32         | 1280                | 5120         | 16        | 632M           |

**Notation:** ViT-Base/16 means Base model with 16×16 patches

---

### 7.2 Mathematical Architecture Breakdown

#### **Input Processing**

Given an input image: $x \in \mathbb{R}^{H \times W \times C}$

**Step 1: Create Patches**
```
Image size: 224×224×3 (Height × Width × Channels)
Patch size: 16×16
Number of patches: N = (224/16) × (224/16) = 14 × 14 = 196 patches
Each patch: 16 × 16 × 3 = 768 values
```

**Step 2: Linear Projection**
$$x_p = [x_p^1; x_p^2; ...; x_p^N] \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

Flatten each patch and project to embedding dimension D:
$$z_0 = [x_{class}; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_{pos}$$

Where:
- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the learnable projection matrix
- $E_{pos} \in \mathbb{R}^{(N+1) \times D}$ is the positional embedding
- $x_{class}$ is the learnable [CLS] token

**Example Calculation:**
```python
# For ViT-Base/16
patch_values = 768  # 16×16×3
embedding_dim = 768
projection = Linear(768, 768)  # Project patch to embedding space
```

---

### 7.3 Transformer Encoder Architecture

#### **Single Transformer Block**

```
Input: z_ℓ (embeddings from previous layer)

1. Layer Normalization (LN)
2. Multi-Head Self-Attention (MSA)
3. Residual Connection
4. Layer Normalization (LN)
5. Multi-Layer Perceptron (MLP)
6. Residual Connection

Output: z_(ℓ+1)
```

**Mathematical Formulation:**

$$z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$$

$$z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell$$

#### **Multi-Head Self-Attention Detail**

For each attention head:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Example Calculation (ViT-Base):**
```
Number of heads: 12
Embedding dimension: 768
Dimension per head: 768/12 = 64

For each patch embedding:
1. Split into 12 heads of size 64
2. Compute Q, K, V for each head
3. Apply scaled dot-product attention
4. Concatenate all heads
5. Project back to 768 dimensions
```

---

### 7.4 Step-by-Step Implementation Example

#### **Complete Forward Pass**

Let's trace a single image through ViT-Base/16:

**Input:** RGB image of size 224×224×3

**Step 1: Patch Extraction**
```python
# Pseudo-code
image = torch.randn(1, 3, 224, 224)  # Batch=1, C=3, H=224, W=224
patches = image.unfold(2, 16, 16).unfold(3, 16, 16)
# Result: (1, 3, 14, 14, 16, 16)
patches = patches.reshape(1, 196, 768)  # 196 patches, each 768-dim
```

**Step 2: Add CLS Token & Position Embeddings**
```python
cls_token = nn.Parameter(torch.randn(1, 1, 768))
pos_embed = nn.Parameter(torch.randn(1, 197, 768))  # 196 patches + 1 CLS

# Prepend CLS token
x = torch.cat([cls_token, patches], dim=1)  # (1, 197, 768)

# Add positional embeddings
x = x + pos_embed  # (1, 197, 768)
```

**Step 3: Through 12 Transformer Layers**
```python
for layer in range(12):
    # Layer Norm + Multi-Head Attention
    attn_out = multihead_attention(layer_norm(x)) + x
    
    # Layer Norm + MLP
    x = mlp(layer_norm(attn_out)) + attn_out
```

**Step 4: Classification**
```python
# Extract CLS token representation
cls_output = x[:, 0]  # (1, 768)

# Classification head
logits = classifier_head(layer_norm(cls_output))  # (1, num_classes)
```

---

### 7.5 Performance Benchmarks

#### **ImageNet-1K Results (Top-1 Accuracy)**

| **Model** | **Pre-training Data** | **Accuracy** | **Parameters** | **Throughput** |
|-----------|----------------------|--------------|----------------|----------------|
| ResNet-50 | ImageNet-1K          | 76.5%        | 25M            | Fast           |
| ResNet-152| ImageNet-1K          | 78.3%        | 60M            | Medium         |
| ViT-B/16  | ImageNet-1K          | 77.9%        | 86M            | Medium         |
| ViT-B/16  | ImageNet-21K         | 84.0%        | 86M            | Medium         |
| ViT-L/16  | ImageNet-21K         | 85.3%        | 307M           | Slower         |
| ViT-H/14  | JFT-300M             | 88.5%        | 632M           | Very Slow      |

**Key Observation:** ViT requires large-scale pre-training to outperform CNNs!

---

### 7.6 Implementation Considerations

#### **Memory Requirements**

Self-attention complexity: $O(N^2 \cdot D)$ where N = number of patches

**Example (ViT-Base/16 on 224×224 image):**
```
Number of patches: 196
Attention matrix per head: 196 × 196 = 38,416 values
With 12 heads: 461,000 values per layer
With 12 layers: ~5.5M attention values total
```

**Memory grows quadratically with image resolution!**

#### **Training Configuration**

**Typical Setup:**
```python
# Optimizer
optimizer = AdamW(lr=0.001, weight_decay=0.1)

# Learning rate schedule
warmup_steps = 10000
lr_schedule = cosine_decay_with_warmup()

# Augmentation
transforms = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ColorJitter(),
    RandAugment(n=2, m=9),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Training
batch_size = 4096  # Distributed across many GPUs
epochs = 300
```

#### **Pre-training vs Fine-tuning**

**Pre-training (Large Dataset):**
- Dataset: ImageNet-21K or JFT-300M
- Duration: 7-14 days on TPU v3-256
- Learning rate: 0.001
- Weight decay: 0.1

**Fine-tuning (Downstream Task):**
- Remove pre-trained head, add new classifier
- Higher resolution possible (e.g., 384×384)
- Fewer epochs: 10-20
- Lower learning rate: 0.0001
- Batch size: 512

---

### 7.7 Practical Code Example

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(3, embed_dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x: (B, 3, 224, 224)
        B = x.shape[0]
        
        # Patch embedding: (B, 768, 14, 14)
        x = self.patch_embed(x)
        
        # Flatten: (B, 768, 196) -> (B, 196, 768)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token: (B, 197, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification from CLS token
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x

# Usage
model = VisionTransformer()
image = torch.randn(1, 3, 224, 224)
output = model(image)  # (1, 1000)
```

---

### 7.8 Key Implementation Insights

✅ **Patch Size Matters:** Smaller patches (14×14) → Better accuracy but slower  
✅ **Pre-training is Critical:** ViT needs 10M+ images to outperform CNNs  
✅ **Position Embeddings:** Both 1D and 2D work, learnable vs fixed (similar results)  
✅ **Resolution Flexibility:** Can fine-tune at higher resolution than pre-training  
✅ **Hybrid Models:** Combining CNN stem with ViT can improve small-data performance  

---

# Diffusion-based Models

## 1. What Are Diffusion Models?

**Definition:** Generative AI models that create images, videos, or audio by learning to remove noise step-by-step. They transform random static into realistic outputs through gradual refinement.

**Key Idea:** Instead of generating content instantly, diffusion models build it progressively—like developing a photograph from fog to a clear image.

---

## 2. How They Work (Simple Explanation)

### Training Phase: Learn to Remove Noise

1. **Start with a real image** (e.g., a photo of a dog)
2. **Add noise gradually** until it becomes random static
3. **Train a model to reverse this process** - predict and remove the noise at each step

### Generation Phase: Create New Images

1. **Start with random noise** (complete static)
2. **Apply the learned denoising** step-by-step (typically 20-50 steps)
3. **Result:** A brand new, realistic image!

**Visual Flow:**
```
Training:  Real Image → Add Noise → Add More Noise → Pure Noise
           
Generation: Pure Noise → Remove Noise → Remove More Noise → New Image
```

---

## 3. Real-World Examples

| **Application** | **Popular Tools** |
|-----------------|-------------------|
| **Text-to-Image** | DALL-E 2, Midjourney, Stable Diffusion |
| **Text-to-Video** | Sora, Runway Gen-2 |
| **Image Editing** | Adobe Firefly, Photoshop AI |
| **Audio Generation** | AudioLDM |

**Example Prompt:** "A cat astronaut floating in space" → Model generates a realistic image in ~30 seconds

---

## 4. Why They're Powerful

✅ **High Quality:** Produces extremely realistic and detailed outputs  
✅ **Controllable:** Guide generation with text prompts (e.g., "make it sunset, oil painting style")  
✅ **Diverse:** Same prompt creates different variations each time  
✅ **Stable Training:** Easier to train than older methods like GANs  

---

## 5. Simple Analogy

**Like a sculptor revealing a statue:**
- **Traditional models:** Try to carve the statue in one attempt (often messy)
- **Diffusion models:** Start with rough stone, gradually refine details step by step (smoother, better quality)

Or think of it as **developing a Polaroid photo:** The image slowly appears from white fog to a clear picture.

---

## 6. Key Takeaway

Diffusion models revolutionized AI-generated content by using a **"gradual refinement"** approach. They power most modern text-to-image tools and represent the current state-of-the-art in generative AI.

**Trade-off:** Higher quality but slower generation (20-50 steps vs 1 step in older models)

---

## 7. Visual Explanations

### Auto-Regressive vs Diffusion Models

**Key Concept:** LLMs are Auto-Regressive models which predict next word one after another

![alt text](image-3.png)

*This image likely compares Auto-Regressive generation (LLMs) with Diffusion generation. Auto-regressive models generate sequentially (word-by-word or token-by-token), while diffusion models generate the entire output iteratively through denoising. Shows the fundamental difference in generation approaches.*

---

### Diffusion Process Visualization

![alt text](image-4.png)

*This image demonstrates the forward and reverse diffusion process visually. Shows how a clear image gradually becomes noise (forward process) and how the model learns to reverse this—transforming noise back into a clear image (reverse process). Each step shows the intermediate states of the image.*

---

### Diffusion Model Architecture & Training

![alt text](image-5.png)

*This image illustrates the diffusion model architecture components: the noise scheduler, the U-Net denoising network, timestep embeddings, and how conditioning (like text prompts) is integrated. Also shows the training loop where the model learns to predict and remove noise at different timesteps.*

![alt text](image-6.png)

