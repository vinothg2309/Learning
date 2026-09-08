05_DEPLOYMENT_AND_OPERATIONS.md
# Deployment & Observability: Kubernetes, GitLab CI/CD, OpenTelemetry, Cost Tracking

## Table of Contents
- [Overview](#overview)
- [Part 1: Kubernetes Architecture](#part-1-kubernetes-architecture)
  - [Core Components](#core-components)
- [Part 2: Observability & Tracing](#part-2-observability--tracing)
  - [OpenTelemetry Integration](#opentelemetry-integration)
  - [Grafana Dashboard](#grafana-dashboard)
- [Part 3: Cost Tracking](#part-3-cost-tracking)
- [Part 4: SLOs & On-Call Runbooks](#part-4-slos--on-call-runbooks)
- [Part 5: GitLab CI/CD](#part-5-gitlab-cicd)
- [Interview Questions](#interview-questions)
- [Next: Interview Q&A](#next-interview-qa)

---

## Overview

The platform must be deployed on **Presight private cloud** with:
1. **Full observability** - Every request traced
2. **Cost transparency** - Token usage → cost per agent
3. **Reliability** - SLOs, incident response, on-call runbooks
4. **Security** - Audit logging, RBAC, secrets management

---

## Part 1: Kubernetes Architecture

### Core Components

```yaml
# presight-agent-platform/values.yaml
# Helm chart for agent platform

apiVersion: v1
kind: Namespace
metadata:
  name: presight-agents

---

# 1. Model Serving Tier
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-serving
  namespace: presight-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      component: model-serving
  template:
    metadata:
      labels:
        component: model-serving
    spec:
      nodeSelector:
        gpu-type: nvidia-h100
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: component
                  operator: In
                  values:
                  - model-serving
              topologyKey: kubernetes.io/hostname
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.3
        args:
          - --model=meta-llama/Llama-2-70b-chat-hf
          - --tensor-parallel-size=8
          - --gpu-memory-utilization=0.9
          - --enable-prefix-caching
        resources:
          requests:
            nvidia.com/gpu: "8"
            memory: "200Gi"
            cpu: "16"
          limits:
            nvidia.com/gpu: "8"
            memory: "200Gi"
            cpu: "16"
        ports:
        - name: http
          containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 5
        env:
        - name: VLLM_DISABLE_CUSTOM_ALL_REDUCE
          value: "1"

---

# 2. Agent Orchestration Layer
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
  namespace: presight-agents
spec:
  replicas: 5
  selector:
    matchLabels:
      component: orchestrator
  template:
    metadata:
      labels:
        component: orchestrator
    spec:
      serviceAccountName: agent-orchestrator
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: orchestrator
        image: presight/agent-orchestrator:v0.1
        ports:
        - name: http
          containerPort: 5000
        - name: metrics
          containerPort: 8888
        env:
        - name: VLLM_ENDPOINT
          value: http://vllm-serving:8000
        - name: POSTGRES_HOST
          valueFrom:
            configMapKeyRef:
              name: platform-config
              key: postgres-host
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: platform-secrets
              key: postgres-user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: platform-secrets
              key: postgres-password
        - name: VAULT_ADDR
          valueFrom:
            configMapKeyRef:
              name: platform-config
              key: vault-addr
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: http://otel-collector:4317
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /etc/presight
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: agent-orchestrator-config

---

# 3. Vector Database (Qdrant)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: presight-agents
spec:
  serviceName: qdrant
  replicas: 3
  selector:
    matchLabels:
      component: vector-db
  template:
    metadata:
      labels:
        component: vector-db
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - name: http
          containerPort: 6333
        - name: grpc
          containerPort: 6334
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
            storage: "100Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi

---

# 4. PostgreSQL (Metadata, Cost Tracking)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: presight-agents
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 500Gi

---

# 5. OpenTelemetry Collector
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: presight-agents
spec:
  replicas: 2
  selector:
    matchLabels:
      component: observability
  template:
    metadata:
      labels:
        component: observability
    spec:
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:latest
        ports:
        - name: otlp-grpc
          containerPort: 4317
        - name: metrics
          containerPort: 8888
        env:
        - name: OTLP_EXPORTER_ENABLED
          value: "true"
        - name: METRICS_EXPORTER
          value: "prometheus"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "2Gi"

---

# 6. Prometheus & Grafana
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: presight-agents
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'agent-orchestrator'
      static_configs:
      - targets: ['agent-orchestrator:8888']
    - job_name: 'vllm'
      static_configs:
      - targets: ['vllm-serving:8000']
```

---

## Part 2: Observability & Tracing with OpenTelemetry

### Overview: What is OpenTelemetry?

**OpenTelemetry (OTel)** is an **open-source, vendor-neutral standard** for collecting and exporting telemetry data (traces, metrics, logs) from applications. Instead of binding your code to specific vendors (Datadog SDK, Splunk SDK, etc.), you instrument once using OTel and can send data to any backend without code changes.

**Core benefit:** Decouple application instrumentation from backend choice.

---

## 1. How OpenTelemetry is Integrated at Enterprises

### Concept Explanation

**The Problem Without OTel:**
Enterprises typically have multiple observability vendors (Datadog for APM, Splunk for logs, Prometheus for metrics). If you hardcode vendor SDKs directly in your applications, you create vendor lock-in:
- Want to switch from Datadog to New Relic? → Rewrite all instrumentation code
- Want to use multiple vendors? → Import N different SDKs in every service

**The Solution with OTel:**
Instrument your application once with the OpenTelemetry SDK (vendor-agnostic). The SDK sends data to an OTel Collector, which acts as a gateway that can route/transform/enrich data before sending to any backend.

**Architecture Flow:**
```
Application (instrumented with OTel SDK)
    ↓ (OTLP protocol: gRPC or HTTP)
OTel Collector (data router, processor, gateway)
    ↓ (Backend-specific exporters)
Observability Backends:
  - Datadog
  - Splunk
  - Prometheus + Grafana
  - AWS X-Ray
  - New Relic
  - Honeycomb
  - Jaeger
```

**Benefits at Enterprise Scale:**
- ✓ **Vendor independence**: Change backends by updating Collector config, not code
- ✓ **Cost optimization**: Route expensive traces to Datadog, cheap metrics to Prometheus
- ✓ **Standardization**: Single telemetry SDK across 100+ microservices
- ✓ **Future-proof**: When new observability tools emerge, no application changes needed

**Real-World Scenario:**
Year 1: Using Datadog (expensive)
Year 2: Auditing Splunk as alternative
Year 3: Decide to use Splunk for cost savings
→ With OTel: Change 3 lines in Collector config
→ Without OTel: Rewrite instrumentation in 200+ services

### Kubernetes YAML Configuration

The enterprise integration pattern is implemented via the OTel Collector running as a sidecar or service:

```yaml
# deployment.yaml: Application with OTel SDK + Sidecar Collector

apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
  namespace: presight-agents
spec:
  replicas: 5
  selector:
    matchLabels:
      app: agent-orchestrator
  template:
    metadata:
      labels:
        app: agent-orchestrator
    spec:
      containers:
      
      # ====== CONTAINER 1: Application ======
      - name: orchestrator
        image: presight/agent-orchestrator:v0.1
        ports:
        - name: http
          containerPort: 5000
        
        # Environment variables for OTel
        env:
        # All telemetry goes to sidecar collector on localhost
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: http://localhost:4317
        
        # Application will send gRPC data to sidecar
        - name: OTEL_EXPORTER_OTLP_PROTOCOL
          value: grpc
        
        # Service name (used in all telemetry)
        - name: OTEL_SERVICE_NAME
          value: presight-agent-orchestrator
        
        # Environment (dev/staging/prod)
        - name: OTEL_DEPLOYMENT_ENVIRONMENT
          value: production
        
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
      
      # ====== CONTAINER 2: OTel Collector Sidecar ======
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:0.88.0
        
        # Ports for receiving and exporting telemetry
        ports:
        - name: otlp-grpc
          containerPort: 4317  # Receive from app
        - name: otlp-http
          containerPort: 4318  # Alternative HTTP endpoint
        - name: prometheus
          containerPort: 8888  # Prometheus metrics scrape endpoint
        
        # Mount configuration from ConfigMap
        volumeMounts:
        - name: otel-collector-config
          mountPath: /etc/otel/config
          readOnly: true
        
        # Pass configuration file to collector
        args: ["--config=/etc/otel/config/otel-collector-config.yaml"]
        
        # Resource limits for collector
        resources:
          requests:
            cpu: "200m"
            memory: "400Mi"
          limits:
            cpu: "500m"
            memory: "1Gi"
        
        # Health checks for collector
        livenessProbe:
          httpGet:
            path: /
            port: 13133  # Health check port
          initialDelaySeconds: 10
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /
            port: 13133
          initialDelaySeconds: 5
          periodSeconds: 5
        
        # Secrets for backend credentials
        env:
        - name: DATADOG_API_KEY
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: datadog-api-key
        - name: SPLUNK_HEC_TOKEN
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: splunk-hec-token
      
      # ====== ConfigMap: OTel Collector Configuration ======
      volumes:
      - name: otel-collector-config
        configMap:
          name: otel-collector-config

---

# configmap.yaml: OTel Collector Configuration

apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: presight-agents
data:
  otel-collector-config.yaml: |
    # ====== RECEIVERS: Where data comes from ======
    receivers:
      
      # OTLP receiver: Accepts data from applications
      otlp:
        protocols:
          # gRPC protocol (performant, used by Go, Python, etc)
          grpc:
            endpoint: 0.0.0.0:4317
          
          # HTTP protocol (fallback, easier debugging)
          http:
            endpoint: 0.0.0.0:4318
      
      # Prometheus receiver: Scrape metrics from endpoints
      prometheus:
        config:
          scrape_configs:
          - job_name: 'agent-orchestrator'
            static_configs:
            - targets: ['localhost:8888']
    
    # ====== PROCESSORS: Transform/enrich data before export ======
    processors:
      
      # Batch processor: Group data into batches (reduce exports)
      batch:
        send_batch_size: 256      # Send after 256 items
        timeout: 10s              # Or send after 10 seconds
        send_batch_max_size: 512  # Don't exceed 512 items per batch
      
      # Memory limiter: Prevent collector from consuming too much memory
      memory_limiter:
        check_interval: 1s
        limit_mib: 512            # Max 512 MB memory usage
        spike_limit_mib: 128      # Allow 128 MB spike
      
      # Kubernetes metadata: Enrich with pod/node information
      k8sattributes:
        auth_type: "serviceAccount"
        node_from_env_var: KUBE_NODE_NAME
        extract:
          metadata:
            - container.id
            - k8s.pod.name
            - k8s.pod.uid
            - k8s.namespace.name
            - k8s.node.name
          labels:
            - tag_name: app
              key: app
      
      # Sampling: Reduce volume (e.g., sample 10% of traces)
      probabilistic_sampler:
        sampling_percentage: 10  # Keep 10%, discard 90%
    
    # ====== EXPORTERS: Where data goes ======
    exporters:
      
      # Datadog exporter
      datadog:
        api:
          key: "${DATADOG_API_KEY}"  # Read from secret
        hostname_source: "config_or_system"
        tags:
          - "env:production"
          - "service:presight-agent"
      
      # Splunk HEC (HTTP Event Collector)
      splunk_hec:
        token: "${SPLUNK_HEC_TOKEN}"
        endpoint: "https://splunk.company.com:8088"
        source: "otel-collector"
        sourcetype: "_json"
      
      # Prometheus exporter (for metrics only)
      prometheus:
        endpoint: "0.0.0.0:8889"
      
      # OTLP exporter (forward to central collector or backend)
      otlp:
        endpoint: "jaeger-collector.observability:4317"
        tls:
          insecure: false
    
    # ====== SERVICE: Connect receivers → processors → exporters ======
    service:
      
      # Pipelines define data flow
      pipelines:
        
        # Traces pipeline
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch, k8sattributes, probabilistic_sampler]
          exporters: [datadog, splunk_hec, otlp]
        
        # Metrics pipeline
        metrics:
          receivers: [otlp, prometheus]
          processors: [memory_limiter, batch, k8sattributes]
          exporters: [datadog, prometheus]
        
        # Logs pipeline
        logs:
          receivers: [otlp]
          processors: [memory_limiter, batch, k8sattributes]
          exporters: [splunk_hec]
      
      # Enable health check endpoint
      health_checks:
        endpoint: 0.0.0.0:13133

---

# secret.yaml: Store backend credentials securely

apiVersion: v1
kind: Secret
metadata:
  name: observability-secrets
  namespace: presight-agents
type: Opaque
stringData:
  datadog-api-key: "YOUR_DATADOG_API_KEY"
  splunk-hec-token: "YOUR_SPLUNK_HEC_TOKEN"
```

---

## 2. OTel Architecture & Components

### Concept Explanation

**The OTel SDK has 6 main components** that work together to collect and export telemetry:

1. **Resource** - Metadata about the source (service name, version, environment)
2. **TracerProvider** - Factory that creates Tracers
3. **SpanProcessor** - Batches/processes spans before export
4. **Exporter** - Sends data to backend (OTLP, Datadog, Splunk, etc.)
5. **MeterProvider** - Factory that creates Meters for metrics
6. **Meter** - Creates metric instruments (counters, histograms, gauges)

**Data Flow:**
```
Application Code
  ↓
SDK (TracerProvider, Tracer create Spans)
  ↓
SpanProcessor (batches spans)
  ↓
Exporter (converts to backend format)
  ↓
OTLP protocol (gRPC or HTTP)
  ↓
OTel Collector (in sidecar)
  ↓
Backend (Datadog, Splunk, Prometheus, Jaeger)
```

### Kubernetes YAML Configuration

```yaml
# Dockerfile: Build application with OTel SDK

FROM python:3.11-slim

# Install OTel libraries
RUN pip install \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-instrumentation-flask \
    opentelemetry-instrumentation-requests \
    opentelemetry-instrumentation-sqlalchemy

WORKDIR /app
COPY . .

# Application will use OTel on startup
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
ENV OTEL_METRICS_EXPORTER=otlp
ENV OTEL_TRACES_EXPORTER=otlp
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true

CMD ["python", "main.py"]

---

# deployment.yaml: Pod with application and OTel collector

apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
  namespace: presight-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-orchestrator
  template:
    metadata:
      labels:
        app: agent-orchestrator
    spec:
      serviceAccountName: agent-orchestrator
      
      containers:
      
      # Main application container
      - name: orchestrator
        image: presight/agent-orchestrator:v0.1
        imagePullPolicy: IfNotPresent
        
        ports:
        - name: http
          containerPort: 5000
          protocol: TCP
        
        # Environment variables that OTel SDK reads
        env:
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://localhost:4317"
        
        - name: OTEL_EXPORTER_OTLP_PROTOCOL
          value: "grpc"
        
        - name: OTEL_SERVICE_NAME
          value: "agent-orchestrator"
        
        - name: OTEL_RESOURCE_ATTRIBUTES
          value: "deployment.environment=production,service.version=0.1.0"
        
        # Enable auto-instrumentation for libraries
        - name: PYTHONPATH
          value: "/usr/local/lib/python3.11/site-packages:/app"
        
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
      
      # OTel Collector sidecar (receives from app, exports to backends)
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:0.88.0
        
        ports:
        # OTLP receivers
        - name: otlp-grpc
          containerPort: 4317
          protocol: TCP
        - name: otlp-http
          containerPort: 4318
          protocol: TCP
        
        # Prometheus scrape endpoint
        - name: prometheus
          containerPort: 8888
          protocol: TCP
        
        # Health check port
        - name: health
          containerPort: 13133
          protocol: TCP
        
        volumeMounts:
        - name: otel-config
          mountPath: /etc/otel
          readOnly: true
        
        args:
        - "--config=/etc/otel/otel-collector-config.yaml"
        
        env:
        # Read secrets from mounted secret
        - name: DATADOG_API_KEY
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: datadog-api-key
        
        - name: SPLUNK_HEC_TOKEN
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: splunk-hec-token
        
        resources:
          requests:
            cpu: "200m"
            memory: "400Mi"
          limits:
            cpu: "500m"
            memory: "1Gi"
        
        livenessProbe:
          httpGet:
            path: /
            port: health
          initialDelaySeconds: 10
          periodSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /
            port: health
          initialDelaySeconds: 5
          periodSeconds: 5
      
      # Volume with collector config
      volumes:
      - name: otel-config
        configMap:
          name: otel-collector-config

---

# rbac.yaml: Service account for RBAC

apiVersion: v1
kind: ServiceAccount
metadata:
  name: agent-orchestrator
  namespace: presight-agents

---

apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: agent-orchestrator
  namespace: presight-agents
rules:
- apiGroups: [""]
  resources: ["pods", "nodes"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["namespaces"]
  verbs: ["get", "list"]

---

apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: agent-orchestrator
  namespace: presight-agents
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: agent-orchestrator
subjects:
- kind: ServiceAccount
  name: agent-orchestrator
  namespace: presight-agents
```

---

## 3. Vendor Agnosticism: How OTel Decouples from Backends

### Concept Explanation

**The Key Insight:**
Your application code doesn't know or care which backend receives the telemetry. The OTel SDK sends data in a standard format (OTLP protocol) to the OTel Collector. The Collector has plugins (exporters) that translate OTLP to any backend's format.

**This means:**
- Write instrumentation once (OTel SDK)
- Export to Datadog, Splunk, Prometheus, New Relic... all without code changes
- Switch backends by updating Collector config

**Real-World Scenario:**
```
2024 Q1: Using Datadog (expensive at scale)
2024 Q2: Testing Splunk as alternative (running both simultaneously)
2024 Q3: Migration complete - Splunk primary, Datadog secondary
2024 Q4: Removing Datadog exporter

Application code: ZERO changes for entire migration
OTel Collector config: Changed 3 times (minimal edits)
```

### Kubernetes YAML Configuration

```yaml
# SCENARIO 1: Using only Datadog

apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
data:
  otel-collector-config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
    
    processors:
      batch:
        send_batch_size: 256
        timeout: 10s
    
    exporters:
      # Only Datadog exporter
      datadog:
        api:
          key: "${DATADOG_API_KEY}"
    
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [datadog]  # ← Traces go to Datadog
        
        metrics:
          receivers: [otlp]
          processors: [batch]
          exporters: [datadog]  # ← Metrics go to Datadog

---

# SCENARIO 2: Migration - Export to both Datadog AND Splunk

apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
data:
  otel-collector-config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
    
    processors:
      batch:
        send_batch_size: 256
        timeout: 10s
    
    exporters:
      # Datadog exporter
      datadog:
        api:
          key: "${DATADOG_API_KEY}"
      
      # Splunk HEC exporter (new)
      splunk_hec:
        token: "${SPLUNK_HEC_TOKEN}"
        endpoint: "https://splunk.company.com:8088"
    
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [datadog, splunk_hec]  # ← Send to BOTH
        
        metrics:
          receivers: [otlp]
          processors: [batch]
          exporters: [datadog, splunk_hec]  # ← Send to BOTH

---

# SCENARIO 3: Splunk is now primary, Datadog is secondary

apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
data:
  otel-collector-config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
    
    processors:
      batch:
        send_batch_size: 256
        timeout: 10s
    
    # Add sampling processor: Only sample 50% to Datadog (expensive)
    probabilistic_sampler:
      sampling_percentage: 50
    
    exporters:
      datadog:
        api:
          key: "${DATADOG_API_KEY}"
      
      splunk_hec:
        token: "${SPLUNK_HEC_TOKEN}"
        endpoint: "https://splunk.company.com:8088"
    
    service:
      pipelines:
        # Traces: Full volume to Splunk, sampled to Datadog
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [splunk_hec]
        
        traces/datadog:
          receivers: [otlp]
          processors: [batch, probabilistic_sampler]  # Sample 50%
          exporters: [datadog]  # Reduce cost
        
        metrics:
          receivers: [otlp]
          processors: [batch]
          exporters: [splunk_hec]  # Metrics go to Splunk

---

# SCENARIO 4: Datadog removed, Splunk + Prometheus

apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
data:
  otel-collector-config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
    
    processors:
      batch:
        send_batch_size: 256
        timeout: 10s
    
    exporters:
      splunk_hec:
        token: "${SPLUNK_HEC_TOKEN}"
        endpoint: "https://splunk.company.com:8088"
      
      prometheus:
        endpoint: "0.0.0.0:8889"
    
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [splunk_hec]
        
        metrics:
          receivers: [otlp]
          processors: [batch]
          exporters: [prometheus]  # Scrape by Prometheus

---

# Application deployment: UNCHANGED throughout all migrations
# The application doesn't care which backend receives data!

apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
spec:
  template:
    spec:
      containers:
      - name: orchestrator
        image: presight/agent-orchestrator:v0.1
        env:
        # Application points to collector on localhost
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: http://localhost:4317
        # No other backend-specific config needed!
      
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:latest
        volumeMounts:
        - name: otel-config
          mountPath: /etc/otel
```

---

## 4. Capabilities: Tracing, Metrics, Logs with K8s Examples

### Concept Explanation

**OpenTelemetry provides three complementary capabilities:**

1. **Distributed Tracing** - Follow a single request through entire system
2. **Metrics** - Track quantitative measurements (throughput, latency, errors)
3. **Logs** - Detailed event messages, linked to traces

**When to use each:**
- **Trace**: "Why did this incident take 5 seconds to process?"
- **Metric**: "How many incidents are we processing per minute?"
- **Log**: "What were the exact steps taken in incident #12345?"

### Kubernetes YAML Configuration

```yaml
# otel-backend-config.yaml: Configure Collector to handle all three

apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
data:
  otel-collector-config.yaml: |
    
    # ====== RECEIVERS: Accept traces, metrics, and logs ======
    receivers:
      
      # OTLP receiver: Standard OpenTelemetry protocol
      otlp:
        protocols:
          # Receive traces and metrics
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318
      
      # Prometheus receiver: Scrape metrics from endpoints
      prometheus:
        config:
          scrape_configs:
          - job_name: 'agent-metrics'
            static_configs:
            - targets: ['localhost:8888']
      
      # Syslog receiver: Accept syslog-formatted logs
      syslog:
        protocol_config:
          transport: tcp
          address: "0.0.0.0:514"
      
      # Fluent Forward receiver: Accept logs from Fluentd
      fluentforward:
        transport: tcp
        address: 0.0.0.0:24224
    
    # ====== PROCESSORS: Transform and enrich ======
    processors:
      
      # Batch: Group data before export (efficient)
      batch:
        send_batch_size: 256
        timeout: 10s
      
      # K8s attributes: Add pod/node metadata
      k8sattributes:
        auth_type: "serviceAccount"
        extract:
          metadata:
            - k8s.pod.name
            - k8s.pod.uid
            - k8s.namespace.name
            - k8s.node.name
      
      # Sampling: Reduce volume
      probabilistic_sampler:
        sampling_percentage: 10  # Keep 10% of traces
      
      # Span processor: Enrich spans with attributes
      attributes:
        actions:
        - key: environment
          value: production
          action: insert
    
    # ====== EXPORTERS: Send to backends ======
    exporters:
      
      # Export traces and metrics to Datadog
      datadog:
        api:
          key: "${DATADOG_API_KEY}"
        use_compression: true
      
      # Export logs to Splunk
      splunk_hec:
        token: "${SPLUNK_HEC_TOKEN}"
        endpoint: "https://splunk.company.com:8088"
      
      # Export metrics to Prometheus (for scraping)
      prometheus:
        endpoint: "0.0.0.0:8889"
      
      # Export traces to Jaeger (for trace visualization)
      jaeger:
        endpoint: jaeger-collector:14250
        tls:
          insecure: true
    
    # ====== SERVICE: Connect pipelines ======
    service:
      pipelines:
        
        # Traces pipeline: OTLP → Datadog + Jaeger
        traces:
          receivers: [otlp]
          processors: [batch, k8sattributes, probabilistic_sampler]
          exporters: [datadog, jaeger]
        
        # Metrics pipeline: OTLP + Prometheus → Datadog + Prometheus
        metrics:
          receivers: [otlp, prometheus]
          processors: [batch, k8sattributes]
          exporters: [datadog, prometheus]
        
        # Logs pipeline: Syslog + Fluent → Splunk
        logs:
          receivers: [syslog, fluentforward, otlp]
          processors: [batch, k8sattributes]
          exporters: [splunk_hec]

---

# Python application: Demonstrate all three capabilities

apiVersion: v1
kind: ConfigMap
metadata:
  name: app-code
data:
  main.py: |
    from opentelemetry import trace, metrics, logs
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    
    # Initialize OpenTelemetry
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(...)
    trace.set_tracer_provider(tracer_provider)
    
    meter_provider = MeterProvider()
    metrics.set_meter_provider(meter_provider)
    
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    
    # Create metric instruments
    request_counter = meter.create_counter("agent.requests.total")
    request_latency = meter.create_histogram("agent.request.duration_ms")
    
    # Distributed trace: Track incident through system
    def process_incident(incident_id: str):
        with tracer.start_as_current_span("process_incident") as span:
            # Add trace attributes
            span.set_attribute("incident.id", incident_id)
            span.set_attribute("incident.severity", "critical")
            
            # Record metric
            request_counter.add(1, {"incident_type": "critical"})
            
            # Sub-span: Analyze
            with tracer.start_as_current_span("analyze") as analyze_span:
                analyze_span.set_attribute("analysis.duration_ms", 500)
                request_latency.record(500, {"step": "analysis"})
            
            # Sub-span: Escalate
            with tracer.start_as_current_span("escalate") as escalate_span:
                escalate_span.set_attribute("escalation.required", True)
                request_latency.record(200, {"step": "escalation"})
            
            # Log event
            logger.info(f"Incident {incident_id} processed", {
                "incident.id": incident_id,
                "trace_id": span.get_span_context().trace_id,
                "duration_ms": 700
            })
    
    if __name__ == "__main__":
        process_incident("SOC-12345")

---

# Service: Expose collector for scraping metrics

apiVersion: v1
kind: Service
metadata:
  name: otel-collector
  namespace: presight-agents
spec:
  selector:
    app: otel-collector
  ports:
  # OTLP gRPC receiver
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
    protocol: TCP
  
  # OTLP HTTP receiver
  - name: otlp-http
    port: 4318
    targetPort: 4318
    protocol: TCP
  
  # Prometheus scrape endpoint
  - name: prometheus
    port: 8888
    targetPort: 8888
    protocol: TCP
  
  # Syslog receiver
  - name: syslog
    port: 514
    targetPort: 514
    protocol: TCP

---

# Prometheus: Scrape OTel Collector metrics

apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
    - job_name: 'otel-collector'
      static_configs:
      - targets: ['otel-collector:8888']
    
    - job_name: 'agent-orchestrator'
      static_configs:
      - targets: ['agent-orchestrator:8888']
```

---

## 5. Docker Container in Kubernetes: Sidecar vs Daemonset vs Centralized

### Concept Explanation

**The Question:**
Where should the OTel Collector run?

**Three Options:**

1. **Sidecar** (in same pod as application)
   - Pros: Simple, self-contained, follows pod lifecycle
   - Cons: Resource overhead per pod, duplicate config

2. **Daemonset** (one per Kubernetes node)
   - Pros: Efficient, single collector per node, lower cost at scale
   - Cons: Node failure = lost telemetry, requires K8s API access

3. **Centralized** (single service in cluster)
   - Pros: Cheapest (3 replicas for entire cluster), single config
   - Cons: Network latency, single point of failure

**Decision Matrix:**
```
<50 pods          → Centralized (simplest, cheapest)
50-500 pods       → Daemonset (balance)
>500 pods         → Daemonset (scale horizontally)
High throughput   → Daemonset (local batching)
```

### Kubernetes YAML Configuration

```yaml
# ====== OPTION 1: SIDECAR (Simplest) ======

apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
  namespace: presight-agents
spec:
  replicas: 5
  selector:
    matchLabels:
      app: agent-orchestrator
  template:
    metadata:
      labels:
        app: agent-orchestrator
    spec:
      containers:
      
      # Main application
      - name: orchestrator
        image: presight/agent-orchestrator:v0.1
        ports:
        - containerPort: 5000
        env:
        # Application sends telemetry to sidecar on localhost
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: http://localhost:4317
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
      
      # OTel Collector sidecar in same pod
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:latest
        ports:
        - name: otlp-grpc
          containerPort: 4317
        - name: prometheus
          containerPort: 8888
        
        volumeMounts:
        - name: otel-config
          mountPath: /etc/otel
        
        args: ["--config=/etc/otel/config.yaml"]
        
        resources:
          requests:
            cpu: "200m"
            memory: "400Mi"
          limits:
            cpu: "500m"
            memory: "1Gi"
      
      volumes:
      - name: otel-config
        configMap:
          name: otel-collector-config

---

# ====== OPTION 2: DAEMONSET (One per node) ======

apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-collector-daemonset
  namespace: presight-agents
spec:
  selector:
    matchLabels:
      app: otel-collector-daemonset
  
  template:
    metadata:
      labels:
        app: otel-collector-daemonset
    spec:
      # Must run on every node
      hostNetwork: true  # Access host network for performance
      hostPID: false
      
      serviceAccountName: otel-collector-daemonset
      
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:latest
        
        # Get node name from environment
        env:
        - name: K8S_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        
        - name: DATADOG_API_KEY
          valueFrom:
            secretKeyRef:
              name: observability-secrets
              key: datadog-api-key
        
        ports:
        # Ports for receiving from pods
        - name: otlp-grpc
          containerPort: 4317
          hostPort: 4317
          protocol: TCP
        
        - name: otlp-http
          containerPort: 4318
          hostPort: 4318
          protocol: TCP
        
        volumeMounts:
        - name: otel-config
          mountPath: /etc/otel
        
        args: ["--config=/etc/otel/config.yaml"]
        
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
        
        livenessProbe:
          httpGet:
            path: /
            port: 13133
          initialDelaySeconds: 10
          periodSeconds: 10
      
      # Don't run collector on master/control-plane nodes
      tolerations:
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
      
      volumes:
      - name: otel-config
        configMap:
          name: otel-collector-config

---

# Application deployment: Point to daemonset (host network)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
spec:
  template:
    spec:
      containers:
      - name: orchestrator
        image: presight/agent-orchestrator:v0.1
        env:
        # Point to daemonset collector on host
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://${K8S_NODE_IP}:4317"
        
        - name: K8S_NODE_IP
          valueFrom:
            fieldRef:
              fieldPath: status.hostIP  # Get node IP

---

# RBAC for daemonset (needs K8s API access)

apiVersion: v1
kind: ServiceAccount
metadata:
  name: otel-collector-daemonset
  namespace: presight-agents

---

apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: otel-collector-daemonset
rules:
- apiGroups: [""]
  resources: ["pods", "nodes"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["namespaces"]
  verbs: ["get", "list"]

---

apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: otel-collector-daemonset
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: otel-collector-daemonset
subjects:
- kind: ServiceAccount
  name: otel-collector-daemonset
  namespace: presight-agents

---

# ====== OPTION 3: CENTRALIZED (Single service) ======

apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector-central
  namespace: presight-agents
spec:
  replicas: 3  # High availability
  selector:
    matchLabels:
      app: otel-collector-central
  
  template:
    metadata:
      labels:
        app: otel-collector-central
    spec:
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-k8s:latest
        
        ports:
        # OTLP receivers (from apps)
        - name: otlp-grpc
          containerPort: 4317
        - name: otlp-http
          containerPort: 4318
        
        # Prometheus scrape
        - name: prometheus
          containerPort: 8888
        
        volumeMounts:
        - name: otel-config
          mountPath: /etc/otel
        
        args: ["--config=/etc/otel/config.yaml"]
        
        resources:
          requests:
            cpu: "1"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "2Gi"
        
        livenessProbe:
          httpGet:
            path: /
            port: 13133
          initialDelaySeconds: 10
          periodSeconds: 10
      
      affinity:
        # Spread replicas across nodes
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - otel-collector-central
              topologyKey: kubernetes.io/hostname
      
      volumes:
      - name: otel-config
        configMap:
          name: otel-collector-config

---

# Service for centralized collector

apiVersion: v1
kind: Service
metadata:
  name: otel-collector-central
  namespace: presight-agents
spec:
  selector:
    app: otel-collector-central
  
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
    protocol: TCP
  
  - name: otlp-http
    port: 4318
    targetPort: 4318
    protocol: TCP
  
  - name: prometheus
    port: 8888
    targetPort: 8888
    protocol: TCP
  
  type: ClusterIP

---

# Application deployment: Point to centralized service

apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
spec:
  template:
    spec:
      containers:
      - name: orchestrator
        image: presight/agent-orchestrator:v0.1
        env:
        # Point to centralized collector service
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://otel-collector-central:4317"
```

---

## Quick Decision Guide: Which Collector Architecture?

```
Questions to ask:

1. How many pods will generate telemetry?
   < 50        → Centralized (simplest)
   50-500      → Daemonset (balance)
   > 500       → Daemonset (scale)

2. What's your throughput (requests/sec)?
   < 1000      → Centralized (cheaper)
   1000-10K    → Daemonset (reduce network)
   > 10K       → Daemonset (local batching)

3. Can you afford a single collector failure?
   Yes         → Centralized (with 3 replicas for HA)
   No          → Daemonset (always available)

4. Do you have node auto-scaling?
   Yes         → Sidecar or Daemonset
   No          → Any option works

PRESIGHT RECOMMENDATION: Sidecar for simplicity (< 50 pods)
                         Daemonset for production scale
```

---

## Part 3: Cost Tracking

### Concept Explanation

**The Problem:**
LLM APIs are expensive. A single request can cost anywhere from $0.001 to $0.10 depending on the model and token count. In a multi-tenant platform, you need to:
- Track who is spending what money
- Enforce budget limits per tenant
- Identify which agents/models are most expensive
- Alert when costs spike unexpectedly

**Why Cost Tracking Matters:**
1. **Prevent runaway costs** - Detect anomalies (e.g., a bug that makes 1000x more API calls)
2. **Multi-tenant billing** - Charge customers accurately
3. **Capacity planning** - Know when to upgrade infrastructure
4. **Optimization** - Identify expensive models and replace with cheaper ones

**Cost Tracking Architecture:**

```
Agent Request
  ↓
Calculate cost (tokens × price_per_token)
  ↓
Log to Database (cost_events table)
  ↓
Budget Check:
  - Running total per tenant per month
  - If >80% spent → send alert
  - If >100% → potentially reject new requests
  ↓
Reporting Dashboard
  - Cost by agent
  - Cost by model
  - Cost by tenant
  - Monthly trends
```

**Key Metrics to Track:**
- **Input tokens** - Cost from prompt (cheap)
- **Output tokens** - Cost from response (expensive)
- **Model used** - GPT-4 costs 10x more than GPT-3.5
- **Cache hit** - Cached requests cost 90% less
- **Tenant ID** - For billing and budgeting
- **Time** - For trend analysis

**Budget Model:**
```
Tenant A monthly budget: $1,000
  ├── SOC Agent: $400 (40%)
  ├── Pentest Agent: $350 (35%)
  └── Code Review Agent: $250 (25%)

When reaching 80% ($800):
  └── Send alert to tenant

When reaching 100% ($1,000):
  └── Option 1: Reject new requests (hard limit)
  └── Option 2: Rate limit (soft limit)
  └── Option 3: Notify customer to increase budget
```

**Cost Calculation Example:**
```
Request A:
  - Model: gpt-4
  - Input: 1,000 tokens @ $0.03/1K = $0.03
  - Output: 500 tokens @ $0.06/1K = $0.03
  - Cache hit: No
  - Total cost: $0.06

Request B (with cache):
  - Same query but hits prompt cache
  - Input tokens (cached): $0.03 × 10% = $0.003
  - Output: 500 tokens @ $0.06/1K = $0.03
  - Total cost: $0.033 (45% cheaper!)
```

### Code Implementation

**1. Database Schema**

```sql
-- Cost tracking tables
CREATE TABLE cost_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    input_tokens INT NOT NULL,
    output_tokens INT NOT NULL,
    cost_usd DECIMAL(10, 6) NOT NULL,
    cache_hit BOOLEAN DEFAULT FALSE,
    latency_ms INT,
    status VARCHAR(50),  -- success, error, timeout
    error_message TEXT
);

-- Index for fast queries
CREATE INDEX idx_cost_events_tenant_timestamp 
ON cost_events(tenant_id, timestamp);

CREATE INDEX idx_cost_events_agent_model 
ON cost_events(agent_name, model_name);

-- Budget configuration per tenant
CREATE TABLE cost_budgets (
    tenant_id VARCHAR(255) PRIMARY KEY,
    monthly_budget_usd DECIMAL(10, 2) NOT NULL,
    alert_threshold_pct INT DEFAULT 80,  -- Alert at 80%
    hard_limit_pct INT DEFAULT 100,      -- Block at 100%
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Track monthly spending for quick lookup
CREATE TABLE cost_monthly_totals (
    tenant_id VARCHAR(255) NOT NULL,
    year_month DATE NOT NULL,  -- 2024-05-01
    total_cost_usd DECIMAL(10, 2),
    request_count INT,
    PRIMARY KEY (tenant_id, year_month)
);

-- Track which models are used per tenant (for optimization)
CREATE TABLE cost_model_usage (
    tenant_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    year_month DATE NOT NULL,
    total_cost_usd DECIMAL(10, 2),
    request_count INT,
    avg_latency_ms INT,
    PRIMARY KEY (tenant_id, model_name, year_month)
);
```

**2. Cost Tracking Middleware (Python)**

```python
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CostCalculator:
    """Calculate cost for different models"""
    
    PRICING = {
        'gpt-4': {
            'input_per_1k': Decimal('0.03'),
            'output_per_1k': Decimal('0.06')
        },
        'gpt-3.5-turbo': {
            'input_per_1k': Decimal('0.0005'),
            'output_per_1k': Decimal('0.0015')
        },
        'claude-opus': {
            'input_per_1k': Decimal('0.015'),
            'output_per_1k': Decimal('0.075')
        },
        'llama-70b': {
            'input_per_1k': Decimal('0.0008'),
            'output_per_1k': Decimal('0.0024')
        }
    }
    
    CACHE_DISCOUNT = Decimal('0.1')  # Cached tokens cost 90% less
    
    @classmethod
    def calculate_cost(
        cls,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool = False
    ) -> Decimal:
        """Calculate total cost for a request"""
        
        if model_name not in cls.PRICING:
            raise ValueError(f"Unknown model: {model_name}")
        
        pricing = cls.PRICING[model_name]
        
        # Calculate input cost
        input_cost = (input_tokens / 1000) * pricing['input_per_1k']
        
        # Apply cache discount to input tokens only
        if cache_hit:
            input_cost *= cls.CACHE_DISCOUNT
        
        # Calculate output cost (no discount)
        output_cost = (output_tokens / 1000) * pricing['output_per_1k']
        
        total = input_cost + output_cost
        return total.quantize(Decimal('0.000001'))


class CostTrackingMiddleware:
    """Track costs at request level"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.calculator = CostCalculator()
    
    def track_request(
        self,
        tenant_id: str,
        agent_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        cache_hit: bool = False,
        status: str = "success"
    ) -> str:
        """
        Track a request and return the request ID.
        Raises alert if budget exceeded.
        """
        
        request_id = str(uuid.uuid4())
        
        # Calculate cost
        cost_usd = self.calculator.calculate_cost(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit
        )
        
        # Log to database
        self.db.execute("""
            INSERT INTO cost_events
            (request_id, tenant_id, agent_name, model_name, 
             input_tokens, output_tokens, cost_usd, cache_hit, 
             latency_ms, status, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            request_id, tenant_id, agent_name, model_name,
            input_tokens, output_tokens, float(cost_usd),
            cache_hit, latency_ms, status, datetime.utcnow()
        ))
        
        # Check budget
        self._check_budget_alert(tenant_id, cost_usd)
        
        logger.info(
            f"Tracked request {request_id}: "
            f"{model_name} | {input_tokens}→{output_tokens} tokens | "
            f"${cost_usd} | Cache hit: {cache_hit}"
        )
        
        return request_id
    
    def _check_budget_alert(self, tenant_id: str, cost_usd: Decimal):
        """Check if tenant is exceeding budget"""
        
        budget_result = self.db.query_one("""
            SELECT monthly_budget_usd, alert_threshold_pct, hard_limit_pct
            FROM cost_budgets
            WHERE tenant_id = %s
        """, (tenant_id,))
        
        if not budget_result:
            return  # No budget set
        
        budget_usd = Decimal(str(budget_result['monthly_budget_usd']))
        alert_threshold = budget_result['alert_threshold_pct']
        hard_limit = budget_result['hard_limit_pct']
        
        # Get current month spending
        current_month_result = self.db.query_one("""
            SELECT COALESCE(SUM(cost_usd), 0) as total
            FROM cost_events
            WHERE tenant_id = %s
              AND DATE_TRUNC('month', timestamp) = DATE_TRUNC('month', NOW())
        """, (tenant_id,))
        
        current_spent = Decimal(str(current_month_result['total']))
        new_total = current_spent + cost_usd
        pct_used = (new_total / budget_usd) * 100
        
        # Alert if approaching budget
        if pct_used >= alert_threshold and pct_used < hard_limit:
            self._send_alert(
                tenant_id=tenant_id,
                alert_type="budget_warning",
                pct_used=float(pct_used),
                spent=float(new_total),
                budget=float(budget_usd),
                severity="warning"
            )
        
        # Block requests if exceeding hard limit
        elif pct_used >= hard_limit:
            self._send_alert(
                tenant_id=tenant_id,
                alert_type="budget_exceeded",
                pct_used=float(pct_used),
                spent=float(new_total),
                budget=float(budget_usd),
                severity="critical"
            )
            raise BudgetExceededError(
                f"Tenant {tenant_id} has exceeded monthly budget "
                f"({pct_used:.1f}%). Current: ${new_total}, Budget: ${budget_usd}"
            )
    
    def _send_alert(self, **kwargs):
        """Send alert to monitoring system"""
        logger.warning(f"Budget alert: {kwargs}")
        # Could also send to Slack, PagerDuty, email, etc.
    
    def get_cost_report(
        self,
        tenant_id: str,
        days: int = 30
    ) -> dict:
        """Get cost breakdown for tenant"""
        
        # Costs by agent
        costs_by_agent = self.db.query("""
            SELECT 
                agent_name,
                COUNT(*) as requests,
                SUM(cost_usd) as total_cost,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
            FROM cost_events
            WHERE tenant_id = %s
              AND timestamp > NOW() - INTERVAL '%d days'
            GROUP BY agent_name
            ORDER BY total_cost DESC
        """, (tenant_id, days))
        
        # Costs by model
        costs_by_model = self.db.query("""
            SELECT 
                model_name,
                COUNT(*) as requests,
                SUM(cost_usd) as total_cost,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens
            FROM cost_events
            WHERE tenant_id = %s
              AND timestamp > NOW() - INTERVAL '%d days'
            GROUP BY model_name
            ORDER BY total_cost DESC
        """, (tenant_id, days))
        
        # Daily costs (for trend analysis)
        daily_costs = self.db.query("""
            SELECT 
                DATE(timestamp) as date,
                SUM(cost_usd) as daily_cost,
                COUNT(*) as request_count
            FROM cost_events
            WHERE tenant_id = %s
              AND timestamp > NOW() - INTERVAL '%d days'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """, (tenant_id, days))
        
        # Total and budget
        total_result = self.db.query_one("""
            SELECT 
                SUM(cost_usd) as total_cost,
                COUNT(*) as total_requests,
                AVG(latency_ms) as avg_latency_ms
            FROM cost_events
            WHERE tenant_id = %s
              AND timestamp > NOW() - INTERVAL '%d days'
        """, (tenant_id, days))
        
        budget_result = self.db.query_one("""
            SELECT monthly_budget_usd
            FROM cost_budgets
            WHERE tenant_id = %s
        """, (tenant_id,))
        
        return {
            'by_agent': [dict(row) for row in costs_by_agent],
            'by_model': [dict(row) for row in costs_by_model],
            'daily_costs': [dict(row) for row in daily_costs],
            'total_cost': float(total_result['total_cost'] or 0),
            'total_requests': total_result['total_requests'] or 0,
            'avg_latency_ms': float(total_result['avg_latency_ms'] or 0),
            'monthly_budget': float(budget_result['monthly_budget_usd'] if budget_result else 0),
            'period_days': days
        }


class BudgetExceededError(Exception):
    """Raised when tenant budget is exceeded"""
    pass
```

**3. Flask Integration (Usage)**

```python
from flask import Flask, jsonify, request
from cost_tracking import CostTrackingMiddleware, BudgetExceededError

app = Flask(__name__)
cost_tracker = CostTrackingMiddleware(db_connection)

@app.route('/api/agents/invoke', methods=['POST'])
def invoke_agent():
    data = request.json
    tenant_id = data['tenant_id']
    agent_name = data['agent_name']
    prompt = data['prompt']
    
    try:
        # Invoke agent
        response = agent_service.invoke(
            agent_name=agent_name,
            prompt=prompt
        )
        
        # Track cost
        request_id = cost_tracker.track_request(
            tenant_id=tenant_id,
            agent_name=agent_name,
            model_name=response['model_used'],
            input_tokens=response['input_tokens'],
            output_tokens=response['output_tokens'],
            latency_ms=response['latency_ms'],
            cache_hit=response.get('cache_hit', False),
            status='success'
        )
        
        return jsonify({
            'request_id': request_id,
            'response': response['output'],
            'cost': response['estimated_cost']
        })
    
    except BudgetExceededError as e:
        # Reject request
        return jsonify({'error': str(e)}), 429
    
    except Exception as e:
        # Track failed request
        cost_tracker.track_request(
            tenant_id=tenant_id,
            agent_name=agent_name,
            model_name='unknown',
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            status='error'
        )
        return jsonify({'error': str(e)}), 500

@app.route('/api/cost-report', methods=['GET'])
def get_cost_report():
    tenant_id = request.args.get('tenant_id')
    days = request.args.get('days', 30, type=int)
    
    report = cost_tracker.get_cost_report(tenant_id, days)
    return jsonify(report)
```

---

## Part 4: SLOs & On-Call Runbooks

### Concept Explanation

**What is an SLO?**

An **SLO (Service Level Objective)** is a measurable target for service reliability. It's a promise to users: "Our service will be available 99.9% of the time" or "Requests will complete in less than 2 seconds 99% of the time."

**Why SLOs Matter:**
1. **Set expectations** - Users know what to expect
2. **Drive priorities** - If you're failing your SLO, that's #1 priority
3. **Enable on-call** - Define when to wake people up (critical SLO failures)
4. **Measure success** - Track improvements over time

**SLO Hierarchy:**

```
SLA (Service Level Agreement) - Contract promise to customers
  ↓
SLO (Service Level Objective) - Internal target
  ├── Availability SLO (99.9%)
  ├── Latency SLO (P99 < 2s)
  ├── Throughput SLO (1000 req/s)
  └── Cost SLO (<$10/hour)
  ↓
SLI (Service Level Indicator) - Actual metric
  ├── Error rate
  ├── P99 latency
  ├── Requests per second
  └── Hourly cost
```

**Example SLOs for Presight Agent Platform:**

| SLO Name | Target | Impact | Alert Threshold |
|----------|--------|--------|-----------------|
| Availability | 99.9% | Errors | >0.1% error rate |
| Latency P99 | <2s | Speed | >2000ms |
| Cost/Hour | <$10 | Budget | >$10 |
| Token Budget | <1M/hr | Usage | >1M tokens/hr |

**On-Call Runbooks:**

An **on-call runbook** is a playbook that tells engineers what to do when an alert fires. It prevents panic and ensures fast resolution.

**Runbook Structure:**
```
Alert Name
├── Severity (Critical, Warning, Info)
├── SLO Impact (which SLO is violated)
├── Symptoms (what does it look like)
├── Quick Diagnosis (first 5 minutes)
├── Response Steps (ordered actions)
├── Escalation Policy (when to call who)
└── Postmortem Checklist (after resolution)
```

### Kubernetes & Prometheus Configuration

**1. Prometheus Rules (Alert Definitions)**

```yaml
# presight-slos.yaml

apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: presight-agent-slos
  namespace: presight-agents
spec:
  groups:
  
  # ====== SLO: AVAILABILITY (99.9%) ======
  - name: presight_availability
    interval: 30s
    rules:
    
    # Alert when error rate exceeds 0.1%
    - alert: AgentUnavailable
      expr: |
        (
          sum(rate(agent_requests_total{status="error"}[5m])) /
          sum(rate(agent_requests_total[5m]))
        ) > 0.001
      for: 5m  # Alert only if sustained for 5 minutes
      labels:
        severity: critical
        slo: availability
      annotations:
        summary: "Agent availability below 99.9%"
        description: |
          Error rate is {{ $value | humanizePercentage }}.
          SLO target: <0.1% error rate
        runbook: "https://wiki.presight.io/runbooks/agent-unavailable"
        dashboard: "https://grafana.presight.io/d/agent-platform"
    
  # ====== SLO: LATENCY (P99 < 2s) ======
  - name: presight_latency
    interval: 30s
    rules:
    
    # Alert when P99 latency exceeds 2 seconds
    - alert: HighLatency
      expr: |
        histogram_quantile(0.99, rate(agent_request_duration_ms[5m])) > 2000
      for: 5m
      labels:
        severity: warning
        slo: latency
      annotations:
        summary: "Agent P99 latency above 2 seconds"
        description: |
          P99 latency is {{ $value }}ms.
          SLO target: <2000ms (2s)
        runbook: "https://wiki.presight.io/runbooks/high-latency"
    
  # ====== SLO: COST (<$10/hour) ======
  - name: presight_cost
    interval: 60s
    rules:
    
    # Alert when hourly cost exceeds $10
    - alert: HighCostRate
      expr: |
        sum(rate(agent_cost_usd_total[1h])) > 10
      for: 10m  # Alert if sustained for 10 minutes
      labels:
        severity: warning
        slo: cost
      annotations:
        summary: "Agent platform costs exceed $10/hour"
        description: |
          Current hourly burn rate: ${{ $value }}.
          SLO target: <$10/hour.
          Monthly impact: ${{ $value * 730 }}
        runbook: "https://wiki.presight.io/runbooks/high-cost"
    
  # ====== SLO: TOKEN BUDGET (<1M/hour) ======
  - name: presight_token_budget
    interval: 60s
    rules:
    
    - alert: HighTokenUsage
      expr: |
        sum(rate(llm_tokens_used_total[1h])) > 1000000
      for: 10m
      labels:
        severity: warning
        slo: token_budget
      annotations:
        summary: "Token usage exceeds budget (>1M/hour)"
        description: |
          Token rate: {{ $value }} tokens/hour.
          SLO target: <1,000,000 tokens/hour
        runbook: "https://wiki.presight.io/runbooks/high-token-usage"
    
  # ====== SLO: CPU/MEMORY PRESSURE ======
  - name: presight_resource_pressure
    interval: 30s
    rules:
    
    - alert: HighCPUUtilization
      expr: |
        sum(rate(container_cpu_usage_seconds_total[5m])) by (pod) > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.pod }} CPU utilization >90%"
        description: "Scale horizontally or optimize code"
    
    - alert: HighMemoryUtilization
      expr: |
        sum(container_memory_working_set_bytes) by (pod) /
        sum(container_spec_memory_limit_bytes) by (pod) > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.pod }} memory utilization >90%"
        description: "May cause OOM kill. Scale or increase memory limits."
```

**2. Prometheus Alerts Configuration**

```yaml
# alertmanager-config.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: presight-agents
data:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    
    # Route alerts to appropriate handlers
    route:
      receiver: 'default'
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s        # Wait 10s before sending alert
      group_interval: 10s    # Wait 10s before sending additional
      repeat_interval: 12h   # Repeat after 12 hours
      
      # Critical alerts go to PagerDuty
      routes:
      - match:
          severity: critical
        receiver: 'pagerduty-critical'
        group_wait: 0s       # Send immediately (no grouping)
        repeat_interval: 1h
      
      # Warning alerts go to Slack
      - match:
          severity: warning
        receiver: 'slack-warnings'
        repeat_interval: 4h
    
    receivers:
    
    # Default receiver (informational)
    - name: 'default'
      slack_configs:
      - channel: '#presight-alerts'
        title: 'Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    
    # Critical alerts → PagerDuty → on-call engineer's phone
    - name: 'pagerduty-critical'
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}'
        details:
          firing: '{{ range .Alerts.Firing }}{{ .Labels.instance }} {{ end }}'
      slack_configs:
      - channel: '#presight-critical'
        title: '🚨 CRITICAL ALERT'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    
    # Warning alerts → Slack
    - name: 'slack-warnings'
      slack_configs:
      - channel: '#presight-alerts'
        title: '⚠️ {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    
    inhibit_rules:
    # Don't alert on high latency if service is down
    - source_match:
        severity: critical
        alertname: AgentUnavailable
      target_match:
        severity: warning
        alertname: HighLatency
      equal: ['cluster', 'service']
```

**3. On-Call Runbooks (Markdown)**

```markdown
# Presight Agent Platform On-Call Runbooks

## Alert: AgentUnavailable

**Status Page:** https://status.presight.io  
**Dashboard:** https://grafana.presight.io/d/agent-platform  
**SLO Impact:** Availability <99.9% (0.1% error rate exceeded)

### 📊 Understanding the Alert

**Trigger:** Error rate exceeds 0.1% for >5 minutes

**Examples of what breaks:**
- API returns 500 errors
- Requests timeout
- Database connection pool exhausted
- Model serving pod crashed

### 🔍 Diagnosis (First 5 minutes)

**1. Check current status**
```bash
# Check error rate
kubectl get prometheusrule -n presight-agents

# Quick health check
curl https://api.presight.io/health
```

**2. Which component is failing?**
```bash
# Check orchestrator logs
kubectl logs -n presight-agents deployment/agent-orchestrator --tail=50

# Check model serving
kubectl logs -n presight-agents deployment/vllm-serving --tail=50

# Check if pods are crashing
kubectl get pods -n presight-agents
kubectl describe pod <pod-name> -n presight-agents
```

**3. Dashboard: Check "Presight Agent Platform" in Grafana**
- **Error Rate panel**: Which agent is failing?
- **P50/P99 latency**: Is it slow or completely broken?
- **Pod Restarts**: Are pods crashing?
- **Resource Usage**: Is CPU/memory maxed out?

### 🔧 Response Steps (Do these in order)

**Step 1: Restart orchestrator (5 min fix)**
```bash
kubectl rollout restart deployment/agent-orchestrator -n presight-agents

# Wait for rollout
kubectl rollout status deployment/agent-orchestrator -n presight-agents

# Verify error rate dropped
# (Check Grafana dashboard)
```

**Step 2: If still failing, restart model serving (10 min fix)**
```bash
kubectl rollout restart deployment/vllm-serving -n presight-agents

# This will cause ~30s downtime (models reload)
# Error rate should drop to <0.1%
```

**Step 3: Check resource constraints (if restarts don't help)**
```bash
# Check node capacity
kubectl top nodes -n presight-agents
kubectl describe nodes

# Check pod resource usage
kubectl top pods -n presight-agents

# If >90% CPU/memory: scale up
kubectl scale deployment agent-orchestrator --replicas=10 -n presight-agents
```

**Step 4: Check database connections**
```bash
# Connect to PostgreSQL
kubectl exec -it postgres-0 -n presight-agents -- psql -U presight

# Check active connections
SELECT datname, count(*) FROM pg_stat_activity GROUP BY datname;

# If >100 connections: likely pool exhausted
# Solution: Restart orchestrator (step 1)
```

### 📋 Escalation Policy

| Time Elapsed | Action |
|---|---|
| 0-5 min | Try restart steps above |
| 5-15 min | Reach out to #presight-oncall |
| 15-30 min | Page platform lead (@platform-on-call) |
| 30+ min | Incident commander (exec) |

**On-Call Contact Info:**
- Slack: #presight-oncall
- PagerDuty: https://presight.pagerduty.com
- Platform Lead: @john-platform-lead (Slack)

### 📝 Post-Resolution Checklist

- [ ] Verify error rate is <0.1%
- [ ] Check logs for root cause
- [ ] File incident in Jira (Ops team)
- [ ] Post in #incidents: "AgentUnavailable resolved at <time>"
- [ ] 24-hour postmortem (if P1 severity)

---

## Alert: HighLatency

**SLO Impact:** P99 latency >2 seconds (SLO target: <2s)

### 🔍 Diagnosis

```bash
# Is it just one agent slow, or all?
kubectl logs deployment/agent-orchestrator -n presight-agents | grep latency

# Check model serving load (GPU utilization)
kubectl top pods -n presight-agents deployment/vllm-serving
nvidia-smi  # Check GPU memory

# Check database query latency
SELECT query, mean_time FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
```

### 🔧 Response Steps

**Step 1: Check GPU utilization**
```bash
kubectl exec -it vllm-serving-0 -n presight-agents -- nvidia-smi
```

- **GPU util >90%?** → Scale up vLLM pods
- **GPU util <70%?** → Problem elsewhere (database, network)

**Step 2: Scale vLLM if needed**
```bash
kubectl scale deployment vllm-serving --replicas=5 -n presight-agents
```

**Step 3: Check database query performance**
```bash
# Enable query log
ALTER SYSTEM SET log_min_duration_statement = 1000;
SELECT pg_reload_conf();

# Find slow queries
SELECT query, calls, mean_time 
FROM pg_stat_statements 
WHERE mean_time > 500 
ORDER BY mean_time DESC;
```

**Step 4: Consider model downgrade (temporary)**
```bash
# Switch to faster model (gpt-3.5 instead of gpt-4)
kubectl set env deployment/agent-orchestrator DEFAULT_MODEL=gpt-3.5-turbo -n presight-agents
```

---

## Alert: HighCostRate

**SLO Impact:** Hourly burn rate exceeds $10

### 🔍 Diagnosis

```bash
# Check cost report
kubectl exec -it orchestrator-0 -n presight-agents -- python -c "
from cost_tracking import CostTrackingMiddleware
tracker = CostTrackingMiddleware(db)
print(tracker.get_cost_report('*', days=1))
"

# Check which model is expensive
SELECT model_name, SUM(cost_usd) as total_cost 
FROM cost_events 
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY model_name
ORDER BY total_cost DESC;
```

### 🔧 Response Steps

**Step 1: Identify expensive model**
- If gpt-4 is >80% of cost → switch to gpt-3.5-turbo
- If specific agent is expensive → check for bugs

**Step 2: Switch model**
```bash
kubectl set env deployment/agent-orchestrator DEFAULT_MODEL=gpt-3.5-turbo -n presight-agents
```

**Step 3: Check for runaway loops**
- Is a single request making 100+ API calls?
- Check logs for "token limit exceeded" errors

**Step 4: Check cache hit ratio**
- Low cache hits = buying expensive tokens twice
- Increase TTL for prompt cache

---

## Alert: HighTokenUsage

**SLO Impact:** Token budget exceeded (>1M tokens/hour)

### 🔍 Diagnosis

```bash
# Which agent is using most tokens?
SELECT agent_name, SUM(input_tokens + output_tokens) as total_tokens
FROM cost_events
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY agent_name
ORDER BY total_tokens DESC;
```

### 🔧 Response

1. Identify high-volume agent
2. Check if there's a bug (e.g., looping)
3. Rate-limit that agent temporarily
4. Investigate in postmortem
```

**4. Runbook as ConfigMap**

```yaml
# runbooks-configmap.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: presight-runbooks
  namespace: presight-agents
data:
  agent-unavailable.md: |
    # Alert: AgentUnavailable
    ...
    (content above)
  
  high-latency.md: |
    # Alert: HighLatency
    ...
    (content above)
  
  high-cost.md: |
    # Alert: HighCostRate
    ...
    (content above)
```

---

## Summary: SLOs & Runbooks

**SLO Targets for Presight:**

| Objective | Target | Alert | On-Call? |
|-----------|--------|-------|----------|
| Availability | 99.9% | Error rate >0.1% | Yes |
| Latency P99 | <2s | >2000ms | Warning only |
| Cost/Hour | <$10 | >$10 | No |
| Token Budget | <1M/hr | >1M | No |

**Runbook Golden Rules:**
1. ✓ **Step-by-step**: No guessing or complex decisions
2. ✓ **Include commands**: Copy-paste ready
3. ✓ **Diagnosis first**: Know what's broken before fixing
4. ✓ **Escalation policy**: When to call humans
5. ✓ **Postmortem checklist**: Learn from incidents

---

## Part 5: GitLab CI/CD

```yaml
# .gitlab-ci.yml

stages:
  - build
  - test
  - evaluate
  - deploy

variables:
  DOCKER_REGISTRY: "docker.presight.internal"
  KUBE_NAMESPACE: "presight-agents"

build_agent_image:
  stage: build
  image: docker:latest
  script:
    - docker build -t $DOCKER_REGISTRY/agent-orchestrator:$CI_COMMIT_SHA .
    - docker push $DOCKER_REGISTRY/agent-orchestrator:$CI_COMMIT_SHA
  only:
    - main
    - merge_requests

unit_tests:
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest tests/unit --cov=src --cov-report=xml
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

evaluation:
  stage: evaluate
  script:
    - python presight/evaluation/run_evaluation.py
      --golden-dataset presight/golden_datasets/soc_golden.json
  artifacts:
    paths:
      - presight/results/evaluation_report.json
  allow_failure: false

regression_test:
  stage: evaluate
  script:
    - python presight/evaluation/regression_test.py
      --current-results presight/results/evaluation_report.json
      --baseline-branch main
  allow_failure: false  # Fail on regressions
  only:
    - merge_requests

deploy_staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context presight/staging
    - helm upgrade --install agent-platform ./helm
      --namespace $KUBE_NAMESPACE
      --values helm/values-staging.yaml
      --set image.tag=$CI_COMMIT_SHA
  environment:
    name: staging
    kubernetes:
      namespace: $KUBE_NAMESPACE
  only:
    - main

deploy_production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context presight/production
    - helm upgrade --install agent-platform ./helm
      --namespace $KUBE_NAMESPACE
      --values helm/values-prod.yaml
      --set image.tag=$CI_COMMIT_SHA
      --wait
      --timeout 5m
  environment:
    name: production
    kubernetes:
      namespace: $KUBE_NAMESPACE
  only:
    - tags
  when: manual  # Manual approval required
```

---

## Interview Questions

1. **Design observability for an agent platform.**
   - Answer: OpenTelemetry tracing, metrics per node, Grafana dashboards, alerts

2. **How do you track costs in a multi-tenant platform?**
   - Answer: Per-request logging, budget enforcement, dashboards per tenant

3. **What SLOs would you set for agent platform?**
   - Answer: 99.9% availability, P99 latency < 2s, cost < $10/hour

4. **Design incident response for agent failures.**
   - Answer: Alerting → diagnosis → runbooks → escalation

---

## Next: Interview Q&A

See `06_INTERVIEW_QA.md` for 50+ technical questions and model answers.