tensorflow_deep_learning_questions.md
# TensorFlow Deep Learning Interview Questions
## For Senior Engineers (13+ Years Experience) - MAANG Level

---

## Part 1: Conceptual Questions (20)

### 1. TensorFlow Graph Execution and Optimization
**Q:** Explain the difference between static graph execution (TensorFlow 1.x) and eager execution (TensorFlow 2.x). How does `@tf.function` bridge these paradigms, and what are the trade-offs in terms of performance, debugging, and AutoGraph limitations?

**Expected Answer Points:**
- Static graphs provide better optimization opportunities (constant folding, CSE, pruning)
- Eager execution enables imperative programming and easier debugging
- `@tf.function` traces Python code to build computational graphs
- AutoGraph converts Python control flow to TensorFlow ops
- Trade-offs: compilation overhead vs execution speed, debugging complexity
- Input signature considerations for polymorphic function handling

---

### 2. Custom Training Loops and GradientTape
**Q:** When would you implement a custom training loop using `tf.GradientTape` instead of using `model.fit()`? Explain how gradient accumulation works and how you would implement it for large batch training on memory-constrained GPUs.

**Expected Answer Points:**
- Custom metrics, loss functions, or training logic not supported by fit()
- Multi-GPU custom distribution strategies
- Gradient accumulation: sum gradients over multiple mini-batches before update
- Implementation: maintain running gradient variable, reset after optimizer step
- Memory efficiency: enables effective larger batch sizes
- Gradient clipping and norm monitoring

---

### 3. Mixed Precision Training
**Q:** Explain TensorFlow's mixed precision training strategy. How does loss scaling prevent gradient underflow, and what are the trade-offs between dynamic vs static loss scaling? How would you debug NaN losses in mixed precision training?

**Expected Answer Points:**
- FP16 for faster computation, FP32 for numerical stability
- Loss scaling multiplies loss to prevent gradient underflow in FP16
- Dynamic loss scaling adjusts scale factor based on gradient overflow detection
- Static loss scaling uses fixed multiplier
- Debugging: check for large activations, gradient monitoring, selective FP32 ops
- Performance gains: 2-3x speedup on Tensor Cores (V100/A100)

---

### 4. TensorFlow Distribution Strategies
**Q:** Compare and contrast `MirroredStrategy`, `MultiWorkerMirroredStrategy`, `TPUStrategy`, and `ParameterServerStrategy`. When would you choose each, and what are the implications for gradient aggregation and variable synchronization?

**Expected Answer Points:**
- MirroredStrategy: single machine, multiple GPUs, all-reduce for gradients
- MultiWorkerMirroredStrategy: multi-machine, ring all-reduce or NCCL
- TPUStrategy: for TPU pods, specialized collective ops
- ParameterServerStrategy: asynchronous training, parameter servers hold variables
- Synchronous vs asynchronous updates trade-offs
- Bandwidth, fault tolerance, and scalability considerations

---

### 5. TensorFlow SavedModel Format
**Q:** Explain the structure of TensorFlow's SavedModel format. How does it differ from HDF5 checkpoints? What are concrete functions, signatures, and how would you optimize a SavedModel for inference serving?

**Expected Answer Points:**
- SavedModel: language-agnostic, contains MetaGraphDef, variables, assets
- HDF5: weights only, requires model definition
- Concrete functions: traced tf.function with specific input signatures
- Signatures: named entry points for serving (predict, serve, etc.)
- Optimization: constant folding, quantization, pruning, TensorRT integration
- Versioning support for model management

---

### 6. Custom Layers and Gradient Computation
**Q:** When creating a custom TensorFlow layer, explain the difference between `build()`, `call()`, and `compute_output_shape()`. How would you implement a layer with non-differentiable operations or custom gradients using `@tf.custom_gradient`?

**Expected Answer Points:**
- build(): create weights lazily based on input shape
- call(): forward pass logic
- compute_output_shape(): for symbolic shape inference
- @tf.custom_gradient: override backward pass
- Use cases: straight-through estimators, quantization, specialized backprop
- get_config() for serialization

---

### 7. TensorFlow Data Pipeline Optimization
**Q:** Explain the performance optimization techniques in `tf.data` API. What is the difference between `interleave()`, `parallel_interleave()`, and `map()` with `num_parallel_calls=tf.data.AUTOTUNE`? How would you diagnose and fix data pipeline bottlenecks?

**Expected Answer Points:**
- Prefetching: overlap data preprocessing with model execution
- Parallel mapping: distribute map function across CPU cores
- Interleave: read from multiple files in parallel
- AUTOTUNE: dynamically adjusts parallelism
- Profiling with TensorBoard, tf.data.experimental.stats_dataset
- Caching, batching order, vectorized map functions
- TFDS sharding and determinism

---

### 8. TensorFlow XLA (Accelerated Linear Algebra)
**Q:** What is XLA and how does it optimize TensorFlow computations? Explain fusion, constant folding, and target-specific code generation. What are the limitations and when should you enable `jit_compile=True`?

**Expected Answer Points:**
- XLA: optimizing compiler for linear algebra
- Fusion: combines multiple ops into single kernel (reduces memory bandwidth)
- Constant folding: pre-compute constants at compile time
- Target-specific: generates optimized code for CPU/GPU/TPU
- Limitations: limited op support, compilation overhead
- Best for: model training loops, inference, repeated computations

---

### 9. TensorFlow Model Optimization Toolkit
**Q:** Compare post-training quantization, quantization-aware training (QAT), and pruning in TensorFlow Model Optimization Toolkit. How does QAT simulate quantization during training, and what are fake quantization nodes?

**Expected Answer Points:**
- Post-training: calibrate activation ranges, weight quantization after training
- QAT: insert fake quant nodes during training to simulate quantization
- Fake quant: quantize then dequantize to model quantization effects
- Pruning: structured vs unstructured, magnitude-based weight removal
- Trade-offs: accuracy vs size/speed
- Representative dataset for calibration

---

### 10. TensorFlow Serving and Model Versioning
**Q:** Explain TensorFlow Serving's architecture. How does it handle model versioning, A/B testing, and canary deployments? What are the differences between gRPC and REST APIs in TF Serving?

**Expected Answer Points:**
- Serving architecture: Manager, Loader, Servable
- Version policy: serve latest, serve specific, serve all
- A/B testing: traffic splitting between versions
- Canary: gradual rollout with monitoring
- gRPC: binary protocol, faster, supports batching
- REST: easier integration, human-readable
- Batching configuration for throughput optimization

---

### 11. Gradient Checkpointing and Memory Management
**Q:** Explain gradient checkpointing (rematerialization) in TensorFlow. How does it trade computation for memory? When would you use it, and how does it interact with TensorFlow's memory allocator?

**Expected Answer Points:**
- Recompute activations during backward pass instead of storing
- Reduces memory footprint for deep networks
- tf.recompute_grad or manual implementation
- Use cases: very deep networks, limited GPU memory
- 20-30% compute overhead for 10x memory reduction
- BFC allocator, memory fragmentation, pre-allocation strategies

---

### 12. TensorFlow Probability and Bayesian Deep Learning
**Q:** How would you implement a Bayesian neural network using TensorFlow Probability? Explain variational inference, reparameterization trick, and how TFP's `DenseVariational` layer works.

**Expected Answer Points:**
- Weight distributions instead of point estimates
- Variational inference: approximate intractable posterior
- Reparameterization: μ + σ * ε for gradient flow
- DenseVariational: learns distribution parameters
- ELBO loss: reconstruction + KL divergence
- Epistemic vs aleatoric uncertainty

---

### 13. TensorFlow Profiler and Performance Analysis
**Q:** What metrics do you analyze using TensorFlow Profiler? Explain device placement, op execution time, and how to identify memory bottlenecks. How would you optimize a model showing low GPU utilization?

**Expected Answer Points:**
- Trace viewer: timeline of op execution
- GPU utilization: kernel launch overhead, data transfer
- Memory profile: peak usage, fragmentation
- Input pipeline analysis: identify data loading bottlenecks
- Low GPU utilization fixes: increase batch size, optimize data pipeline, reduce CPU preprocessing
- Step time breakdown: input, compute, output

---

### 14. Custom Training Step with Multiple Optimizers
**Q:** How would you implement a training step using different optimizers for different parts of a model (e.g., generator and discriminator in GANs)? Explain how gradient computation and variable tracking works with multiple optimizers.

**Expected Answer Points:**
- Multiple GradientTape contexts or single tape with multiple minimize calls
- Separate trainable_variables lists for each model component
- Generator optimizer applies to generator vars only
- Discriminator optimizer applies to discriminator vars
- Careful gradient computation ordering
- Learning rate scheduling per optimizer

---

### 15. TensorFlow Lite Conversion and Mobile Optimization
**Q:** What are the key differences between TensorFlow and TensorFlow Lite? Explain the conversion process, delegate selection (GPU/NNAPI/CoreML), and how to handle unsupported ops during conversion.

**Expected Answer Points:**
- TFLite: lightweight runtime for mobile/embedded
- Conversion: to FlatBuffer format, op fusion, quantization
- Delegates: hardware-specific acceleration (GPU delegate, NNAPI, Hexagon)
- Unsupported ops: use Select TF ops or implement custom ops
- Model optimization: dynamic range/int8/float16 quantization
- Representative dataset for full integer quantization

---

### 16. TensorFlow Gradient Accumulation for Transformers
**Q:** When training large transformer models that don't fit in GPU memory, how would you implement gradient accumulation in TensorFlow? What are the implications for batch normalization and learning rate scaling?

**Expected Answer Points:**
- Accumulate gradients over N mini-batches, update once
- Scale learning rate by √N or linearly
- Batch norm: uses mini-batch statistics, not effective batch
- Layer norm preferred for gradient accumulation
- Implementation: persistent GradientTape or manual accumulation
- Memory: stores only gradients, not intermediate activations

---

### 17. TensorFlow Model Garden and Transfer Learning
**Q:** Explain the architecture of TensorFlow Model Garden. How would you fine-tune a pre-trained model (e.g., EfficientNet) for a custom task? What are the best practices for learning rate, layer freezing, and progressive unfreezing?

**Expected Answer Points:**
- Model Garden: official implementations, pre-trained weights
- Transfer learning: freeze base, train head first
- Progressive unfreezing: gradually unfreeze deeper layers
- Learning rate: lower for pre-trained layers (discriminative fine-tuning)
- Data augmentation importance for small datasets
- Feature extraction vs fine-tuning trade-offs

---

### 18. TensorFlow Distributed Training Communication Patterns
**Q:** Explain the communication patterns in distributed training: all-reduce, ring all-reduce, and parameter server. How does NCCL optimize GPU communication, and what are the bandwidth implications at scale?

**Expected Answer Points:**
- All-reduce: aggregate gradients across workers
- Ring all-reduce: bandwidth-optimal, 2(N-1) transfers
- Parameter server: centralized weight storage, async updates
- NCCL: NVIDIA library for collective ops, GPU-direct RDMA
- Bandwidth: becomes bottleneck at scale (100s of GPUs)
- Gradient compression, local SGD for reducing communication

---

### 19. TensorFlow Keras Functional API vs Subclassing
**Q:** When would you use Keras Sequential, Functional API, or Model Subclassing? What are the trade-offs in terms of flexibility, serialization, and multi-input/multi-output support?

**Expected Answer Points:**
- Sequential: simple linear stacks, limited flexibility
- Functional: multi-input/output, shared layers, easy serialization
- Subclassing: maximum flexibility, dynamic computation graphs
- Functional API: graph of layers, can be inspected and modified
- Subclassing: imperative style, harder to serialize
- Model saving: Functional better for SavedModel export

---

### 20. TensorFlow Autograph and Control Flow
**Q:** How does AutoGraph convert Python control flow (if, while, for) into TensorFlow operations? What are the limitations when using dynamic conditions or Python built-ins inside @tf.function?

**Expected Answer Points:**
- Converts Python control flow to tf.cond, tf.while_loop
- Static analysis of Python AST
- Limitations: dynamic shapes, Python side effects, external state
- tf.TensorShape vs dynamic shape handling
- Concrete function tracing and retracing behavior
- input_signature for controlling polymorphism

---

## Part 2: Scenario-Based Questions (10)

### Scenario 1: Production Model Performance Degradation
**Q:** Your TensorFlow model deployed via TF Serving shows 3x slower inference time in production compared to development. The model uses a ResNet50 backbone. What systematic approach would you take to diagnose and fix this issue?

**Expected Approach:**
1. Compare environments: TF versions, hardware (GPU vs CPU), batch sizes
2. Profile production inference: TF Profiler, latency breakdown
3. Check batching configuration in TF Serving (max_batch_size, batch_timeout)
4. Verify model optimization: XLA compilation, TensorRT integration
5. Network latency: gRPC vs REST overhead, payload size
6. Input preprocessing: check if preprocessing is duplicated client-side and server-side
7. Resource contention: check GPU sharing, CPU throttling
8. Model version: ensure correct model artifact deployed
9. Enable request logging and trace slow requests
10. Consider: quantization, pruning, distillation for smaller model

---

### Scenario 2: Out-of-Memory During Training
**Q:** You're training a Vision Transformer (ViT) on 4x V100 GPUs (16GB each) with batch size 32 per GPU. Training crashes with OOM after 100 steps. Inference works fine. What strategies would you implement to fix this?

**Expected Solutions:**
1. Enable mixed precision training (FP16): 2x memory reduction
2. Gradient checkpointing: recompute activations during backward pass
3. Reduce batch size, use gradient accumulation (effective batch size maintained)
4. Optimize data pipeline: reduce prefetch buffer, remove unnecessary caching
5. Check for memory leaks: clear session between runs, check growing tensors
6. Profile memory usage: identify peak usage ops
7. Model architecture: reduce hidden dimensions, number of heads
8. Use `tf.function` to optimize graph memory
9. Enable TensorFlow memory growth: allow_growth=True
10. Consider: model parallelism, pipeline parallelism for very large models

---

### Scenario 3: Multi-Modal Model Architecture
**Q:** Design a TensorFlow architecture for a multi-modal model that combines image (ResNet50), text (BERT), and tabular features for e-commerce product classification. How would you handle different input types, fusion strategies, and ensure efficient training?

**Expected Design:**
1. **Architecture:**
   - Image branch: pre-trained ResNet50, global pooling → 512-d
   - Text branch: pre-trained BERT, [CLS] token → 768-d
   - Tabular: dense layers with batch norm → 256-d
   - Fusion: concatenation → shared dense layers → classification head

2. **Implementation Details:**
   - Functional API for multi-input model
   - Separate learning rates per branch (discriminative fine-tuning)
   - Feature-level fusion vs decision-level fusion
   - Attention-based fusion for learning modality importance

3. **Training Strategy:**
   - Pre-train branches separately, then joint fine-tuning
   - Modality dropout: randomly drop modalities for robustness
   - Balanced batch sampling across categories
   - Mixed precision for faster training

4. **Data Pipeline:**
   - tf.data: parallel processing per modality
   - Efficient image decoding (JPEG optimization)
   - BERT tokenization in preprocessing
   - Prefetching and caching

---

### Scenario 4: Real-time Inference Optimization
**Q:** You need to deploy a TensorFlow object detection model (YOLOv5) for real-time video processing (30 FPS) on edge devices (Jetson Nano with 4GB RAM). Current inference time is 150ms. How would you optimize this?

**Expected Optimization Strategy:**
1. **Model Optimization:**
   - TensorFlow Lite conversion with int8 quantization
   - Pruning: remove 30-40% weights
   - Knowledge distillation: smaller student model
   - Reduce input resolution (640 → 416)
   - Reduce number of detection layers

2. **Runtime Optimization:**
   - GPU delegate for Jetson
   - Batch processing if latency allows
   - Model caching and warmup
   - Reduce confidence threshold for faster NMS

3. **Pipeline Optimization:**
   - Asynchronous inference: process while capturing next frame
   - Skip frames: process every 2nd frame
   - Frame preprocessing optimization
   - Use TensorRT for Jetson-specific optimization

4. **Measurement:**
   - Profile each component: decode, preprocess, inference, postprocess
   - Target: 33ms per frame (30 FPS)
   - Monitor GPU/CPU utilization

---

### Scenario 5: Handling Class Imbalance in Medical Imaging
**Q:** You're building a TensorFlow model for medical image classification with severe class imbalance (1:100 ratio between disease and normal cases). The model achieves 99% accuracy but misses most disease cases. Design a comprehensive solution.

**Expected Solution:**
1. **Metrics:**
   - Replace accuracy with F1-score, precision-recall AUC
   - Per-class metrics, confusion matrix
   - Focus on recall for positive class

2. **Data Level:**
   - Oversampling minority class (SMOTE for images)
   - Undersampling majority class
   - Synthetic data augmentation for minority class
   - Mixup/CutMix augmentation

3. **Algorithm Level:**
   - Class weights in loss function: inverse frequency weighting
   - Focal loss: down-weight easy examples
   - Two-stage training: pre-train on balanced data

4. **Architecture:**
   - Attention mechanisms to focus on relevant regions
   - Transfer learning from similar medical datasets
   - Ensemble multiple models with different sampling

5. **Validation:**
   - Stratified k-fold cross-validation
   - Hold-out test set maintaining distribution
   - Clinical relevance: prioritize false negatives vs false positives

---

### Scenario 6: Distributed Training Failure Recovery
**Q:** Your multi-worker distributed training job (8 workers, ParameterServerStrategy) frequently fails due to worker preemption on cloud VMs. Training takes 3 days. How would you implement fault tolerance and minimize restart overhead?

**Expected Implementation:**
1. **Checkpointing Strategy:**
   - Frequent checkpoints (every 10-15 minutes)
   - Save to distributed filesystem (GCS, S3)
   - Atomic checkpoint writes
   - Keep last N checkpoints for rollback

2. **Fault Tolerance:**
   - tf.data snapshot: cache preprocessed data
   - Worker discovery and dynamic rescheduling
   - Graceful degradation: continue with fewer workers
   - Health checks and automatic restart

3. **Optimization:**
   - Incremental checkpoints (save only changed variables)
   - Asynchronous checkpoint writing
   - Checkpoint sharding across workers
   - Resume from latest checkpoint automatically

4. **Monitoring:**
   - Worker health monitoring
   - Checkpoint validation
   - Training metrics persistence
   - Alert on failures

5. **Alternative:**
   - Use spot instances strategically
   - MirroredStrategy on fewer reliable nodes
   - Gradient accumulation to reduce communication

---

### Scenario 7: Custom Loss Function for Ranking
**Q:** Implement a TensorFlow custom loss function for a learning-to-rank problem (e.g., search result ranking). The loss should consider pairwise ordering and work with `model.fit()`. How would you ensure numerical stability and efficient gradient computation?

**Expected Implementation:**
```python
class PairwiseRankingLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, name='pairwise_ranking_loss'):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        # y_true: relevance scores [batch_size, num_items]
        # y_pred: predicted scores [batch_size, num_items]

        # Create pairs: positive items should rank higher than negative
        # Expand dimensions for pairwise comparison
        y_pred_i = tf.expand_dims(y_pred, axis=2)  # [B, N, 1]
        y_pred_j = tf.expand_dims(y_pred, axis=1)  # [B, 1, N]

        y_true_i = tf.expand_dims(y_true, axis=2)
        y_true_j = tf.expand_dims(y_true, axis=1)

        # Create mask where i should rank higher than j
        relevance_mask = tf.cast(y_true_i > y_true_j, tf.float32)

        # Pairwise hinge loss
        loss = tf.maximum(0.0, self.margin - (y_pred_i - y_pred_j))

        # Apply mask and average
        masked_loss = loss * relevance_mask
        return tf.reduce_sum(masked_loss) / (tf.reduce_sum(relevance_mask) + 1e-10)
```

**Key Considerations:**
- Numerical stability: epsilon for division by zero
- Gradient flow: use differentiable operations
- Efficient computation: vectorized pairwise comparisons
- Memory: quadratic in number of items, consider sampling
- Alternative: ListNet, LambdaRank implementations

---

### Scenario 8: A/B Testing ML Models
**Q:** You need to A/B test two TensorFlow models in production: model_v1 (current) and model_v2 (new). Design a system using TF Serving that allows gradual rollout, performance monitoring, and automatic rollback if model_v2 underperforms.

**Expected Design:**
1. **TF Serving Setup:**
   - Deploy both models with different version labels
   - Configure model policy to serve both versions
   - Use model metadata for version identification

2. **Traffic Splitting:**
   - Load balancer level: route X% to v1, Y% to v2
   - Feature flag service: control split percentage
   - Gradual rollout: 5% → 10% → 25% → 50% → 100%
   - User-level sticky assignment (hash-based)

3. **Monitoring:**
   - Latency percentiles (p50, p95, p99) per model
   - Prediction distribution drift
   - Business metrics: CTR, conversion rate
   - Error rates and exceptions
   - Resource utilization per model

4. **Automatic Rollback:**
   - Define SLOs: latency < 100ms, error rate < 0.1%
   - Real-time monitoring with alerting
   - Automatic traffic shift if SLO violated
   - Circuit breaker pattern

5. **Analysis:**
   - Statistical significance testing (t-test, chi-square)
   - Minimum sample size calculation
   - Segment analysis: performance by user cohort
   - Shadow mode: log predictions without serving

---

### Scenario 9: Continual Learning Pipeline
**Q:** Design a TensorFlow-based continual learning system for a recommendation model that needs to incorporate new user interactions daily without forgetting old patterns. Address catastrophic forgetting and model drift.

**Expected Design:**
1. **Architecture:**
   - Base model: pre-trained on historical data
   - Adapter layers: task-specific fine-tuning
   - Elastic weight consolidation (EWC): protect important weights
   - Progressive neural networks: new columns for new data

2. **Training Strategy:**
   - Incremental learning: train on new data + sampled old data
   - Replay buffer: store representative samples from old data
   - Knowledge distillation: old model as teacher
   - Multi-task learning: auxiliary tasks for regularization

3. **Implementation:**
```python
class ContinualLearningModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.fisher_information = {}  # For EWC

    def ewc_loss(self, old_params, fisher, lambda_ewc=0.5):
        loss = 0
        for var in self.trainable_variables:
            if var.name in fisher:
                loss += tf.reduce_sum(
                    fisher[var.name] * tf.square(var - old_params[var.name])
                )
        return lambda_ewc * loss
```

4. **Monitoring:**
   - Performance on historical test set (forgetting metric)
   - Performance on new data (plasticity metric)
   - Model drift detection: feature distribution monitoring
   - Embedding drift: cosine similarity over time

5. **Operational:**
   - Daily training jobs on new data
   - Checkpoint management: keep model history
   - A/B test new model before full deployment
   - Fallback to previous version if drift detected

---

### Scenario 10: Multi-Tenant Model Serving
**Q:** You're building a TensorFlow Serving infrastructure for multiple clients, each with custom fine-tuned models based on a common backbone. Design a system that efficiently serves 100+ models with low latency, optimal resource utilization, and cost efficiency.

**Expected Architecture:**
1. **Model Organization:**
   - Shared backbone (base model): loaded once in memory
   - Client-specific heads: loaded on-demand
   - Model registry: maps client_id → model_version
   - Model inheritance: avoid duplicating backbone weights

2. **Serving Strategy:**
   - Model batching: group requests by model version
   - Dynamic batching: batch requests within time window
   - Model multiplexing: serve multiple models per GPU
   - GPU memory sharing: careful memory management

3. **Optimization:**
   - Model caching: LRU cache for frequently used models
   - Lazy loading: load models on first request
   - Model unloading: evict unused models after timeout
   - Quantization: reduce memory footprint per model

4. **Implementation:**
   - TF Serving with multiple model configs
   - Sidecar pattern: routing service + TF Serving
   - Request router: based on client_id, route to appropriate model
   - Kubernetes deployment: auto-scaling based on load

5. **Cost Optimization:**
   - Cold start optimization: keep popular models warm
   - Resource pooling: share GPU across tenants
   - Priority queuing: SLA-based prioritization
   - Monitoring: per-client usage tracking for billing

**Code Snippet:**
```python
# Model config for multi-tenant serving
model_config_list = {
    'config': [
        {
            'name': 'base_model',
            'base_path': '/models/base',
            'model_platform': 'tensorflow'
        },
        {
            'name': 'client_1_model',
            'base_path': '/models/client_1',
            'model_platform': 'tensorflow'
        },
        # ... more client models
    ]
}
```

---

## Answer Key Themes for Evaluators

### Conceptual Questions - Look for:
- Deep understanding of TensorFlow internals
- Performance optimization awareness
- Production deployment experience
- Trade-off analysis (accuracy vs speed, memory vs compute)
- Distributed training knowledge
- Modern practices (TF 2.x, eager execution, AutoGraph)

### Scenario Questions - Look for:
- Systematic problem-solving approach
- Profiling and debugging methodology
- Real-world production experience
- Cost and resource optimization
- Monitoring and observability
- Fault tolerance and reliability
- End-to-end system design thinking

### Red Flags:
- Only theoretical knowledge without practical application
- Outdated practices (TF 1.x session-based code)
- No mention of profiling/monitoring
- Ignoring trade-offs and constraints
- No consideration for production deployment
- Lack of debugging methodology

### Bonus Points:
- Experience with TensorRT, ONNX integration
- Knowledge of quantization techniques
- MLOps practices (model versioning, CI/CD)
- Cost optimization strategies
- Scalability considerations
- Multi-cloud deployment experience
- Open-source contributions to TensorFlow ecosystem

---

**Note:** These questions are designed for senior engineers (13+ years) interviewing at MAANG companies. The expected depth of answers should demonstrate:
1. Hands-on production experience
2. Scale and performance optimization
3. System design capabilities
4. Trade-off analysis
5. Modern best practices (2026)

