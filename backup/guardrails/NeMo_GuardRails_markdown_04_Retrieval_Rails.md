
# NeMo Guardrails: Retrieval Rails (RAG Integration)





## Overview


Retrieval rails control and validate how information is retrieved from knowledge bases in RAG (Retrieval Augmented Generation) systems. They ensure:

Relevant information retrieval

Source validation and citation

Privacy and access control

Quality filtering of retrieved content

Prevention of information leakage




## Types of Retrieval Rails


1. **Knowledge Base Integration**: Connect to vector stores and knowledge bases

2. **Relevance Filtering**: Ensure retrieved content is relevant

3. **Source Attribution**: Track and cite information sources

4. **Access Control**: Enforce permissions on retrieved data

5. **Quality Scoring**: Rank and filter retrieval results

## Setup



# Install additional dependencies


!pip install chromadb sentence-transformers langchain-community faiss-cpu


Output:
Collecting chromadb
Downloading chromadb-1.4.1-cp39-abi3-macosx_11_0_arm64.whl.metadata (7.2 kB)
Collecting sentence-transformers
Downloading sentence_transformers-5.2.2-py3-none-any.whl.metadata (16 kB)
Requirement already satisfied: langchain-community in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (0.4.1)
Collecting faiss-cpu
Downloading faiss_cpu-1.13.2-cp310-abi3-macosx_14_0_arm64.whl.metadata (7.6 kB)
Collecting build>=1.0.3 (from chromadb)
Downloading build-1.4.0-py3-none-any.whl.metadata (5.8 kB)
Requirement already satisfied: pydantic>=1.9 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (2.12.5)
Collecting pybase64>=1.4.1 (from chromadb)
Downloading pybase64-1.4.3-cp310-cp310-macosx_11_0_arm64.whl.metadata (8.7 kB)
Requirement already satisfied: uvicorn>=0.18.3 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from uvicorn[standard]>=0.18.3->chromadb) (0.40.0)
Requirement already satisfied: numpy>=1.22.5 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (2.2.6)
Collecting posthog<6.0.0,>=2.4.0 (from chromadb)
Downloading posthog-5.4.0-py3-none-any.whl.metadata (5.7 kB)
Requirement already satisfied: typing-extensions>=4.5.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (4.15.0)
Requirement already satisfied: onnxruntime>=1.14.1 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (1.23.2)
Collecting opentelemetry-api>=1.2.0 (from chromadb)
Using cached opentelemetry_api-1.39.1-py3-none-any.whl.metadata (1.5 kB)
Collecting opentelemetry-exporter-otlp-proto-grpc>=1.2.0 (from chromadb)
Downloading opentelemetry_exporter_otlp_proto_grpc-1.39.1-py3-none-any.whl.metadata (2.5 kB)
Collecting opentelemetry-sdk>=1.2.0 (from chromadb)
Using cached opentelemetry_sdk-1.39.1-py3-none-any.whl.metadata (1.5 kB)
Requirement already satisfied: tokenizers>=0.13.2 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (0.22.2)
Collecting pypika>=0.48.9 (from chromadb)
Downloading pypika-0.50.0-py2.py3-none-any.whl.metadata (51 kB)
Requirement already satisfied: tqdm>=4.65.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (4.67.1)
Collecting overrides>=7.3.1 (from chromadb)
Downloading overrides-7.7.0-py3-none-any.whl.metadata (5.8 kB)
Collecting importlib-resources (from chromadb)
Downloading importlib_resources-6.5.2-py3-none-any.whl.metadata (3.9 kB)
Collecting grpcio>=1.58.0 (from chromadb)
Downloading grpcio-1.76.0-cp310-cp310-macosx_11_0_universal2.whl.metadata (3.7 kB)
Collecting bcrypt>=4.0.1 (from chromadb)
Downloading bcrypt-5.0.0-cp39-abi3-macosx_10_12_universal2.whl.metadata (10 kB)
Requirement already satisfied: typer>=0.9.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (0.21.1)
Collecting kubernetes>=28.1.0 (from chromadb)
Downloading kubernetes-35.0.0-py2.py3-none-any.whl.metadata (1.7 kB)
Requirement already satisfied: tenacity>=8.2.3 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (9.1.2)
Requirement already satisfied: pyyaml>=6.0.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (6.0.3)
Requirement already satisfied: mmh3>=4.0.1 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (5.2.0)
Requirement already satisfied: orjson>=3.9.12 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (3.11.5)
Requirement already satisfied: httpx>=0.27.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (0.28.1)
Requirement already satisfied: rich>=10.11.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from chromadb) (14.3.1)
Collecting jsonschema>=4.19.0 (from chromadb)
Downloading jsonschema-4.26.0-py3-none-any.whl.metadata (7.6 kB)
Requirement already satisfied: requests<3.0,>=2.7 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from posthog<6.0.0,>=2.4.0->chromadb) (2.32.5)
Requirement already satisfied: six>=1.5 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from posthog<6.0.0,>=2.4.0->chromadb) (1.17.0)
Requirement already satisfied: python-dateutil>=2.2 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from posthog<6.0.0,>=2.4.0->chromadb) (2.9.0.post0)
Collecting backoff>=1.10.0 (from posthog<6.0.0,>=2.4.0->chromadb)
Using cached backoff-2.2.1-py3-none-any.whl.metadata (14 kB)
Requirement already satisfied: distro>=1.5.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from posthog<6.0.0,>=2.4.0->chromadb) (1.9.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from requests<3.0,>=2.7->posthog<6.0.0,>=2.4.0->chromadb) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from requests<3.0,>=2.7->posthog<6.0.0,>=2.4.0->chromadb) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from requests<3.0,>=2.7->posthog<6.0.0,>=2.4.0->chromadb) (2.6.3)
Requirement already satisfied: certifi>=2017.4.17 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from requests<3.0,>=2.7->posthog<6.0.0,>=2.4.0->chromadb) (2026.1.4)
Collecting transformers<6.0.0,>=4.41.0 (from sentence-transformers)
Downloading transformers-5.0.0-py3-none-any.whl.metadata (37 kB)
Requirement already satisfied: huggingface-hub>=0.20.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from sentence-transformers) (1.3.4)
Collecting torch>=1.11.0 (from sentence-transformers)
Downloading torch-2.10.0-cp310-none-macosx_11_0_arm64.whl.metadata (31 kB)
Collecting scikit-learn (from sentence-transformers)
Downloading scikit_learn-1.7.2-cp310-cp310-macosx_12_0_arm64.whl.metadata (11 kB)
Collecting scipy (from sentence-transformers)
Downloading scipy-1.15.3-cp310-cp310-macosx_14_0_arm64.whl.metadata (61 kB)
Requirement already satisfied: filelock in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (3.20.3)
Requirement already satisfied: packaging>=20.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (25.0)
Requirement already satisfied: regex!=2019.12.17 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (2026.1.15)
Requirement already satisfied: typer-slim in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.21.1)
Collecting safetensors>=0.4.3 (from transformers<6.0.0,>=4.41.0->sentence-transformers)
Using cached safetensors-0.7.0-cp38-abi3-macosx_11_0_arm64.whl.metadata (4.1 kB)
Requirement already satisfied: fsspec>=2023.5.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2026.1.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.2.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.2.0)
Requirement already satisfied: shellingham in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.5.4)
Requirement already satisfied: anyio in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from httpx>=0.27.0->chromadb) (4.12.1)
Requirement already satisfied: httpcore==1.* in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from httpx>=0.27.0->chromadb) (1.0.9)
Requirement already satisfied: h11>=0.16 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from httpcore==1.*->httpx>=0.27.0->chromadb) (0.16.0)
Requirement already satisfied: langchain-core<2.0.0,>=1.0.1 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-community) (1.2.7)
Requirement already satisfied: langchain-classic<2.0.0,>=1.0.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-community) (1.0.1)
Requirement already satisfied: SQLAlchemy<3.0.0,>=1.4.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-community) (2.0.46)
Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-community) (3.13.3)
Requirement already satisfied: dataclasses-json<0.7.0,>=0.6.7 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-community) (0.6.7)
Requirement already satisfied: pydantic-settings<3.0.0,>=2.10.1 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-community) (2.12.0)
Requirement already satisfied: langsmith<1.0.0,>=0.1.125 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-community) (0.6.6)
Requirement already satisfied: httpx-sse<1.0.0,>=0.4.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-community) (0.4.3)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (2.6.1)
Requirement already satisfied: aiosignal>=1.4.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (1.4.0)
Requirement already satisfied: async-timeout<6.0,>=4.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (4.0.3)
Requirement already satisfied: attrs>=17.3.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (25.4.0)
Requirement already satisfied: frozenlist>=1.1.1 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (6.7.1)
Requirement already satisfied: propcache>=0.2.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (0.4.1)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain-community) (1.22.0)
Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from dataclasses-json<0.7.0,>=0.6.7->langchain-community) (3.26.2)
Requirement already satisfied: typing-inspect<1,>=0.4.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from dataclasses-json<0.7.0,>=0.6.7->langchain-community) (0.9.0)
Requirement already satisfied: langchain-text-splitters<2.0.0,>=1.1.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-classic<2.0.0,>=1.0.0->langchain-community) (1.1.0)
Requirement already satisfied: jsonpatch<2.0.0,>=1.33.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-core<2.0.0,>=1.0.1->langchain-community) (1.33)
Requirement already satisfied: uuid-utils<1.0,>=0.12.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langchain-core<2.0.0,>=1.0.1->langchain-community) (0.14.0)
Requirement already satisfied: jsonpointer>=1.9 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from jsonpatch<2.0.0,>=1.33.0->langchain-core<2.0.0,>=1.0.1->langchain-community) (3.0.0)
Requirement already satisfied: requests-toolbelt>=1.0.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langsmith<1.0.0,>=0.1.125->langchain-community) (1.0.0)
Requirement already satisfied: zstandard>=0.23.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from langsmith<1.0.0,>=0.1.125->langchain-community) (0.25.0)
Requirement already satisfied: annotated-types>=0.6.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from pydantic>=1.9->chromadb) (0.7.0)
Requirement already satisfied: pydantic-core==2.41.5 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from pydantic>=1.9->chromadb) (2.41.5)
Requirement already satisfied: typing-inspection>=0.4.2 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from pydantic>=1.9->chromadb) (0.4.2)
Requirement already satisfied: python-dotenv>=0.21.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from pydantic-settings<3.0.0,>=2.10.1->langchain-community) (1.2.1)
Requirement already satisfied: mypy-extensions>=0.3.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7.0,>=0.6.7->langchain-community) (1.1.0)
Collecting pyproject_hooks (from build>=1.0.3->chromadb)
Downloading pyproject_hooks-1.2.0-py3-none-any.whl.metadata (1.3 kB)
Collecting tomli>=1.1.0 (from build>=1.0.3->chromadb)
Downloading tomli-2.4.0-py3-none-any.whl.metadata (10 kB)
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=4.19.0->chromadb)
Downloading jsonschema_specifications-2025.9.1-py3-none-any.whl.metadata (2.9 kB)
Collecting referencing>=0.28.4 (from jsonschema>=4.19.0->chromadb)
Downloading referencing-0.37.0-py3-none-any.whl.metadata (2.8 kB)
Collecting rpds-py>=0.25.0 (from jsonschema>=4.19.0->chromadb)
Downloading rpds_py-0.30.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (4.1 kB)
Collecting websocket-client!=0.40.0,!=0.41.*,!=0.42.*,>=0.32.0 (from kubernetes>=28.1.0->chromadb)
Downloading websocket_client-1.9.0-py3-none-any.whl.metadata (8.3 kB)
Collecting requests-oauthlib (from kubernetes>=28.1.0->chromadb)
Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)
Collecting durationpy>=0.7 (from kubernetes>=28.1.0->chromadb)
Downloading durationpy-0.10-py3-none-any.whl.metadata (340 bytes)
Requirement already satisfied: coloredlogs in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from onnxruntime>=1.14.1->chromadb) (15.0.1)
Requirement already satisfied: flatbuffers in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from onnxruntime>=1.14.1->chromadb) (25.12.19)
Requirement already satisfied: protobuf in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from onnxruntime>=1.14.1->chromadb) (6.33.4)
Requirement already satisfied: sympy in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from onnxruntime>=1.14.1->chromadb) (1.14.0)
Collecting importlib-metadata<8.8.0,>=6.0 (from opentelemetry-api>=1.2.0->chromadb)
Downloading importlib_metadata-8.7.1-py3-none-any.whl.metadata (4.7 kB)
Collecting zipp>=3.20 (from importlib-metadata<8.8.0,>=6.0->opentelemetry-api>=1.2.0->chromadb)
Using cached zipp-3.23.0-py3-none-any.whl.metadata (3.6 kB)
Collecting googleapis-common-protos~=1.57 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)
Using cached googleapis_common_protos-1.72.0-py3-none-any.whl.metadata (9.4 kB)
Collecting opentelemetry-exporter-otlp-proto-common==1.39.1 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)
Downloading opentelemetry_exporter_otlp_proto_common-1.39.1-py3-none-any.whl.metadata (1.8 kB)
Collecting opentelemetry-proto==1.39.1 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)
Downloading opentelemetry_proto-1.39.1-py3-none-any.whl.metadata (2.3 kB)
Collecting opentelemetry-semantic-conventions==0.60b1 (from opentelemetry-sdk>=1.2.0->chromadb)
Using cached opentelemetry_semantic_conventions-0.60b1-py3-none-any.whl.metadata (2.4 kB)
Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from rich>=10.11.0->chromadb) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from rich>=10.11.0->chromadb) (2.19.2)
Requirement already satisfied: mdurl~=0.1 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->chromadb) (0.1.2)
Collecting networkx>=2.5.1 (from torch>=1.11.0->sentence-transformers)
Using cached networkx-3.4.2-py3-none-any.whl.metadata (6.3 kB)
Requirement already satisfied: jinja2 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from torch>=1.11.0->sentence-transformers) (3.1.6)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from sympy->onnxruntime>=1.14.1->chromadb) (1.3.0)
Requirement already satisfied: click>=8.0.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from typer>=0.9.0->chromadb) (8.3.1)
Collecting httptools>=0.6.3 (from uvicorn[standard]>=0.18.3->chromadb)
Downloading httptools-0.7.1-cp310-cp310-macosx_11_0_arm64.whl.metadata (3.5 kB)
Collecting uvloop>=0.15.1 (from uvicorn[standard]>=0.18.3->chromadb)
Downloading uvloop-0.22.1-cp310-cp310-macosx_10_9_universal2.whl.metadata (4.9 kB)
Collecting watchfiles>=0.13 (from uvicorn[standard]>=0.18.3->chromadb)
Downloading watchfiles-1.1.1-cp310-cp310-macosx_11_0_arm64.whl.metadata (4.9 kB)
Collecting websockets>=10.4 (from uvicorn[standard]>=0.18.3->chromadb)
Downloading websockets-16.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (6.8 kB)
Requirement already satisfied: exceptiongroup>=1.0.2 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from anyio->httpx>=0.27.0->chromadb) (1.3.1)
Requirement already satisfied: humanfriendly>=9.1 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from coloredlogs->onnxruntime>=1.14.1->chromadb) (10.0)
Requirement already satisfied: MarkupSafe>=2.0 in /Users/vinotganesan/anaconda3/envs/nemo/lib/python3.10/site-packages (from jinja2->torch>=1.11.0->sentence-transformers) (3.0.3)
Collecting oauthlib>=3.0.0 (from requests-oauthlib->kubernetes>=28.1.0->chromadb)
Downloading oauthlib-3.3.1-py3-none-any.whl.metadata (7.9 kB)
Collecting joblib>=1.2.0 (from scikit-learn->sentence-transformers)
Downloading joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn->sentence-transformers)
Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Downloading chromadb-1.4.1-cp39-abi3-macosx_11_0_arm64.whl (19.6 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m19.6/19.6 MB[0m [31m5.1 MB/s[0m  [33m0:00:04[0m eta [36m0:00:01[0mm
[?25hDownloading posthog-5.4.0-py3-none-any.whl (105 kB)
Downloading sentence_transformers-5.2.2-py3-none-any.whl (494 kB)
Downloading transformers-5.0.0-py3-none-any.whl (10.1 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.1/10.1 MB[0m [31m12.5 MB/s[0m  [33m0:00:00[0meta [36m0:00:01[0m
[?25hDownloading faiss_cpu-1.13.2-cp310-abi3-macosx_14_0_arm64.whl (3.5 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.5/3.5 MB[0m [31m11.4 MB/s[0m  [33m0:00:00[0mm0:00:01[0m0:01[0m
[?25hUsing cached backoff-2.2.1-py3-none-any.whl (15 kB)
Downloading bcrypt-5.0.0-cp39-abi3-macosx_10_12_universal2.whl (495 kB)
Downloading build-1.4.0-py3-none-any.whl (24 kB)
Downloading grpcio-1.76.0-cp310-cp310-macosx_11_0_universal2.whl (11.8 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.8/11.8 MB[0m [31m6.2 MB/s[0m  [33m0:00:01[0m eta [36m0:00:01[0mm
[?25hDownloading jsonschema-4.26.0-py3-none-any.whl (90 kB)
Downloading jsonschema_specifications-2025.9.1-py3-none-any.whl (18 kB)
Downloading kubernetes-35.0.0-py2.py3-none-any.whl (2.0 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m10.7 MB/s[0m  [33m0:00:00[0meta [36m0:00:01[0m
[?25hDownloading durationpy-0.10-py3-none-any.whl (3.9 kB)
Using cached opentelemetry_api-1.39.1-py3-none-any.whl (66 kB)
Downloading importlib_metadata-8.7.1-py3-none-any.whl (27 kB)
Downloading opentelemetry_exporter_otlp_proto_grpc-1.39.1-py3-none-any.whl (19 kB)
Downloading opentelemetry_exporter_otlp_proto_common-1.39.1-py3-none-any.whl (18 kB)
Downloading opentelemetry_proto-1.39.1-py3-none-any.whl (72 kB)
Using cached googleapis_common_protos-1.72.0-py3-none-any.whl (297 kB)
Using cached opentelemetry_sdk-1.39.1-py3-none-any.whl (132 kB)
Using cached opentelemetry_semantic_conventions-0.60b1-py3-none-any.whl (219 kB)
Downloading overrides-7.7.0-py3-none-any.whl (17 kB)
Downloading pybase64-1.4.3-cp310-cp310-macosx_11_0_arm64.whl (31 kB)
Downloading pypika-0.50.0-py2.py3-none-any.whl (60 kB)
Downloading referencing-0.37.0-py3-none-any.whl (26 kB)
Downloading rpds_py-0.30.0-cp310-cp310-macosx_11_0_arm64.whl (359 kB)
Using cached safetensors-0.7.0-cp38-abi3-macosx_11_0_arm64.whl (447 kB)
Downloading tomli-2.4.0-py3-none-any.whl (14 kB)
Downloading torch-2.10.0-cp310-none-macosx_11_0_arm64.whl (79.4 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m79.4/79.4 MB[0m [31m6.4 MB/s[0m  [33m0:00:12[0mm0:00:01[0mm00:01[0m
[?25hUsing cached networkx-3.4.2-py3-none-any.whl (1.7 MB)
Downloading httptools-0.7.1-cp310-cp310-macosx_11_0_arm64.whl (109 kB)
Downloading uvloop-0.22.1-cp310-cp310-macosx_10_9_universal2.whl (1.3 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m9.6 MB/s[0m  [33m0:00:00[0m eta [36m0:00:01[0m
[?25hDownloading watchfiles-1.1.1-cp310-cp310-macosx_11_0_arm64.whl (394 kB)
Downloading websocket_client-1.9.0-py3-none-any.whl (82 kB)
Downloading websockets-16.0-cp310-cp310-macosx_11_0_arm64.whl (175 kB)
Using cached zipp-3.23.0-py3-none-any.whl (10 kB)
Downloading importlib_resources-6.5.2-py3-none-any.whl (37 kB)
Downloading pyproject_hooks-1.2.0-py3-none-any.whl (10 kB)
Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Downloading oauthlib-3.3.1-py3-none-any.whl (160 kB)
Downloading scikit_learn-1.7.2-cp310-cp310-macosx_12_0_arm64.whl (8.7 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m8.7/8.7 MB[0m [31m11.8 MB/s[0m  [33m0:00:00[0m eta [36m0:00:01[0m
[?25hDownloading joblib-1.5.3-py3-none-any.whl (309 kB)
Downloading scipy-1.15.3-cp310-cp310-macosx_14_0_arm64.whl (22.4 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m22.4/22.4 MB[0m [31m4.1 MB/s[0m  [33m0:00:05[0m6m0:00:01[0m00:01[0m
[?25hDownloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Installing collected packages: durationpy, zipp, websockets, websocket-client, uvloop, tomli, threadpoolctl, scipy, safetensors, rpds-py, pyproject_hooks, pypika, pybase64, overrides, opentelemetry-proto, oauthlib, networkx, joblib, importlib-resources, httptools, grpcio, googleapis-common-protos, faiss-cpu, bcrypt, backoff, torch, scikit-learn, requests-oauthlib, referencing, posthog, opentelemetry-exporter-otlp-proto-common, importlib-metadata, build, watchfiles, opentelemetry-api, kubernetes, jsonschema-specifications, opentelemetry-semantic-conventions, jsonschema, opentelemetry-sdk, transformers, opentelemetry-exporter-otlp-proto-grpc, sentence-transformers, chromadb
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44/44[0m [chromadb]chromadb]sentence-transformers]ntions]
[1A[2KSuccessfully installed backoff-2.2.1 bcrypt-5.0.0 build-1.4.0 chromadb-1.4.1 durationpy-0.10 faiss-cpu-1.13.2 googleapis-common-protos-1.72.0 grpcio-1.76.0 httptools-0.7.1 importlib-metadata-8.7.1 importlib-resources-6.5.2 joblib-1.5.3 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 kubernetes-35.0.0 networkx-3.4.2 oauthlib-3.3.1 opentelemetry-api-1.39.1 opentelemetry-exporter-otlp-proto-common-1.39.1 opentelemetry-exporter-otlp-proto-grpc-1.39.1 opentelemetry-proto-1.39.1 opentelemetry-sdk-1.39.1 opentelemetry-semantic-conventions-0.60b1 overrides-7.7.0 posthog-5.4.0 pybase64-1.4.3 pypika-0.50.0 pyproject_hooks-1.2.0 referencing-0.37.0 requests-oauthlib-2.0.0 rpds-py-0.30.0 safetensors-0.7.0 scikit-learn-1.7.2 scipy-1.15.3 sentence-transformers-5.2.2 threadpoolctl-3.6.0 tomli-2.4.0 torch-2.10.0 transformers-5.0.0 uvloop-0.22.1 watchfiles-1.1.1 websocket-client-1.9.0 websockets-16.0 zipp-3.23.0



import os

from nemoguardrails import RailsConfig, LLMRails

from langchain_community.vectorstores import Chroma, FAISS

from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

import yaml




# API Configuration


[REDACTED_API_KEY]

BASE_URL = ""



os.environ["OPENAI_API_KEY"] = API_KEY

os.environ["OPENAI_API_BASE"] = BASE_URL


## Example 1: Basic Knowledge Base Integration




Set up a simple vector store and integrate it with NeMo Guardrails.


# Create sample documents for knowledge base


sample_documents = [

Document(

page_content="Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",

metadata={"source": "python_intro.txt", "category": "programming"}

),

Document(

page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",

metadata={"source": "ml_basics.txt", "category": "ai"}

),

Document(

page_content="NeMo Guardrails is a toolkit for adding programmable guardrails to LLM-based conversational systems. It supports input rails, output rails, and retrieval rails.",

metadata={"source": "nemo_docs.txt", "category": "documentation"}

),

Document(

page_content="Vector databases store high-dimensional vectors and enable efficient similarity search. They are essential for RAG applications.",

metadata={"source": "vector_db.txt", "category": "database"}

),

Document(

page_content="LangChain is a framework for developing applications powered by language models. It provides tools for chaining LLM calls and integrating with external data sources.",

metadata={"source": "langchain_intro.txt", "category": "framework"}

)

]



print(f"Created {len(sample_documents)} sample documents")


Output:
Created 5 sample documents




# Create embeddings (using a local model for demo)


from langchain_community.embeddings import HuggingFaceEmbeddings



embeddings = HuggingFaceEmbeddings(

model_name="sentence-transformers/all-MiniLM-L6-v2"

)



print("✓ Embeddings model loaded")


Output:
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 1745.05it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



Output:
✓ Embeddings model loaded




# Create FAISS vector store


vectorstore = FAISS.from_documents(sample_documents, embeddings)




# Save for later use


!mkdir -p configs/rag_basic/kb

vectorstore.save_local("configs/rag_basic/kb")



print("✓ Vector store created and saved")


Output:
✓ Vector store created and saved




# Test retrieval


query = "What is Python?"

results = vectorstore.similarity_search(query, k=2)



print(f"Query: {query}\n")

for i, doc in enumerate(results, 1):

print(f"Result {i}:")

print(f"Content: {doc.page_content}")

print(f"Source: {doc.metadata['source']}\n")


Output:
Query: What is Python?

Result 1:
Content: Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.
Source: python_intro.txt

Result 2:
Content: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.
Source: ml_basics.txt




## Example 2: NeMo Guardrails with Knowledge Base




Configure NeMo to use knowledge base for grounded responses.


# Configuration with knowledge base


kb_config = """

models:

- type: main

engine: openai

model: gpt-4o-mini

parameters:

temperature: 0.0

max_tokens: 256



instructions:

- type: general

content: |

You are a helpful assistant that answers questions based on the provided knowledge base.

Always cite your sources when providing information.

If information is not in the knowledge base, clearly state that.



rails:

retrieval:

flows:

- retrieve from kb

"""



# Colang for retrieval


kb_colang = """

define user ask question

"what is"

"tell me about"

"explain"



define flow retrieve from kb

user ask question

$relevant_docs = execute retrieve_relevant_chunks

bot provide answer

"""



# Custom retrieval action


kb_actions = '''

from nemoguardrails.actions import action

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings



@action(is_system_action=True)

async def retrieve_relevant_chunks(context: dict):

"""Retrieve relevant chunks from knowledge base."""



user_message = context.get("last_user_message", "")



# Load vector store

embeddings = HuggingFaceEmbeddings(

model_name="sentence-transformers/all-MiniLM-L6-v2"

)

vectorstore = FAISS.load_local(

"configs/rag_basic/kb",

embeddings,

allow_dangerous_deserialization=True

)



# Retrieve relevant documents

results = vectorstore.similarity_search_with_score(user_message, k=3)



# Format results

relevant_chunks = []

for doc, score in results:

relevant_chunks.append({

"content": doc.page_content,

"source": doc.metadata.get("source", "unknown"),

"score": float(score)

})



# Add to context for LLM

context["relevant_chunks"] = relevant_chunks



return relevant_chunks

'''



# Save configuration


!mkdir -p configs/rag_basic



with open('configs/rag_basic/config.yml', 'w') as f:

f.write(kb_config)



with open('configs/rag_basic/config.co', 'w') as f:

f.write(kb_colang)



with open('configs/rag_basic/actions.py', 'w') as f:

f.write(kb_actions)



print("✓ RAG configuration saved")


Output:
✓ RAG configuration saved



import nest_asyncio




# Apply nest_asyncio to allow async calls in Jupyter


nest_asyncio.apply()




# Test Example 2: Basic RAG Configuration


print("Testing Basic RAG Configuration...")

print("=" * 70)




# Load the configuration


config_rag = RailsConfig.from_path("configs/rag_basic")

rails_rag = LLMRails(config_rag)



print("✓ RAG Configuration loaded successfully\n")




# Test queries


test_queries = [

"What is Python?",

"Explain machine learning",

"What is NeMo Guardrails?",

"Tell me about LangChain"

]



for idx, query in enumerate(test_queries, 1):

print(f"Test {idx}: {query}")

print("-" * 70)



response = rails_rag.generate(messages=[{"role": "user", "content": query}])

print(f"Bot: {response['content']}")

print()



print("=" * 70)

print("✓ Basic RAG tests completed!")

print("\nNote: Responses should be based on retrieved documents from KB")


Output:
Testing Basic RAG Configuration...
======================================================================
✓ RAG Configuration loaded successfully

Test 1: What is Python?
----------------------------------------------------------------------



Output:
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 2118.03it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.



Output:
Bot:

Test 2: Explain machine learning
----------------------------------------------------------------------



Output:
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 2117.30it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.



Output:
Bot:

Test 3: What is NeMo Guardrails?
----------------------------------------------------------------------



Output:
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 2196.53it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.



Output:
Bot:

Test 4: Tell me about LangChain
----------------------------------------------------------------------



Output:
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 2188.55it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.



Output:
Bot:

======================================================================
✓ Basic RAG tests completed!

Note: Responses should be based on retrieved documents from KB



## Example 3: Relevance Filtering




Filter retrieved documents based on relevance scores.




### Understanding Bot Messages in Colang




In the example below, you'll see two different types of bot message patterns:



1. **Implicit Bot Messages** (LLM-Generated):

- `bot provide grounded answer` - NOT explicitly defined in Colang

- NeMo Guardrails uses the LLM to generate a response based on context

- The retrieved documents are automatically available to the LLM

- More flexible and context-aware



2. **Explicit Bot Messages** (Predefined):

- `bot inform no relevant info` - Defined with a specific message

- Uses the exact text specified in the `define` statement

- Consistent and predictable response



**Why use implicit messages?**

When you write `bot provide grounded answer` without defining it, NeMo allows the LLM to:

Generate natural, contextual responses

Use the retrieved documents from `$docs`

Follow the instructions in config.yml

Adapt the answer to the specific question



This is particularly useful for RAG applications where you want the LLM to synthesize information from retrieved documents rather than using a static template.


# Configuration with relevance filtering


relevance_config = """

models:

- type: main

engine: openai

model: gpt-4o-mini

parameters:

temperature: 0.0

max_tokens: 256



instructions:

- type: general

content: |

You are a helpful assistant.

Only use highly relevant information from the knowledge base.

If no relevant information is found, say so.



rails:

retrieval:

flows:

- retrieve and filter

"""



# Colang for relevance filtering


relevance_colang = """

define flow retrieve and filter

user ask question

$docs = execute retrieve_with_relevance_filter



if $docs.has_relevant

bot provide grounded answer

else

bot inform no relevant info



define bot inform no relevant info

"I don't have relevant information in my knowledge base to answer that question accurately."

"""



# Relevance filtering action


relevance_actions = '''

from nemoguardrails.actions import action

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings



@action(is_system_action=True)

async def retrieve_with_relevance_filter(context: dict):

"""Retrieve documents with relevance filtering."""



user_message = context.get("last_user_message", "")



# Relevance threshold (lower is better for L2 distance)

RELEVANCE_THRESHOLD = 1.0



# Load vector store

embeddings = HuggingFaceEmbeddings(

model_name="sentence-transformers/all-MiniLM-L6-v2"

)

vectorstore = FAISS.load_local(

"configs/rag_basic/kb",

embeddings,

allow_dangerous_deserialization=True

)



# Retrieve with scores

results = vectorstore.similarity_search_with_score(user_message, k=5)



# Filter by relevance

relevant_docs = []

for doc, score in results:

if score < RELEVANCE_THRESHOLD:

relevant_docs.append({

"content": doc.page_content,

"source": doc.metadata.get("source", "unknown"),

"relevance_score": float(score)

})



return {

"documents": relevant_docs,

"has_relevant": len(relevant_docs) > 0,

"count": len(relevant_docs)

}

'''



# Save configuration


!mkdir -p configs/relevance_filtering



with open('configs/relevance_filtering/config.yml', 'w') as f:

f.write(relevance_config)



with open('configs/relevance_filtering/config.co', 'w') as f:

f.write(relevance_colang)



with open('configs/relevance_filtering/actions.py', 'w') as f:

f.write(relevance_actions)




# Copy vector store


!cp -r configs/rag_basic/kb configs/relevance_filtering/



print("✓ Relevance filtering configuration saved")


Output:
✓ Relevance filtering configuration saved




# Test Example 3: Relevance Filtering


print("Testing Relevance Filtering Configuration...")

print("=" * 70)




# Load the configuration


config_relevance = RailsConfig.from_path("configs/relevance_filtering")

rails_relevance = LLMRails(config_relevance)



print("✓ Configuration loaded successfully\n")




# Test 1: Query with relevant information in KB


print("Test 1: Query with RELEVANT information")

print("-" * 70)

query1 = "What is Python?"

response1 = rails_relevance.generate(messages=[{"role": "user", "content": query1}])

print(f"User: {query1}")

print(f"Bot: {response1['content']}")

print()




# Test 2: Query with NO relevant information in KB


print("Test 2: Query with NO relevant information")

print("-" * 70)

query2 = "What is quantum entanglement?"

response2 = rails_relevance.generate(messages=[{"role": "user", "content": query2}])

print(f"User: {query2}")

print(f"Bot: {response2['content']}")

print()




# Test 3: Another relevant query


print("Test 3: Query about machine learning")

print("-" * 70)

query3 = "Tell me about machine learning"

response3 = rails_relevance.generate(messages=[{"role": "user", "content": query3}])

print(f"User: {query3}")

print(f"Bot: {response3['content']}")

print()



print("=" * 70)

print("✓ All tests completed!")

print("\nExpected Behavior:")

print("  - Test 1 & 3: Should provide grounded answers from knowledge base")

print("  - Test 2: Should return 'no relevant information' message")


Output:
Testing Relevance Filtering Configuration...
======================================================================
✓ Configuration loaded successfully

Test 1: Query with RELEVANT information
----------------------------------------------------------------------
User: What is Python?
Bot: Python is a high-level, interpreted programming language known for its readability and simplicity. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It has a large standard library and a vibrant ecosystem of third-party packages, making it suitable for a wide range of applications, from web development and data analysis to artificial intelligence and scientific computing. Python's syntax emphasizes code readability, which helps developers write clear and maintainable code.

Test 2: Query with NO relevant information
----------------------------------------------------------------------
User: What is quantum entanglement?
Bot: Quantum entanglement is a physical phenomenon that occurs when pairs or groups of particles become interconnected in such a way that the quantum state of one particle cannot be described independently of the state of the other(s), even when the particles are separated by large distances. This means that the measurement of one particle's properties (such as its position, momentum, or spin) instantaneously affects the state of the other particle, regardless of the distance between them.

Entanglement is a key feature of quantum mechanics and has been experimentally confirmed in numerous studies. It challenges classical intuitions about the separability of distant objects and has implications for quantum information science, including quantum computing and quantum cryptography. The phenomenon is often illustrated by the thought experiment known as "Einstein-Podolsky-Rosen (EPR) paradox," which highlights the non-locality of quantum mechanics.

Test 3: Query about machine learning
----------------------------------------------------------------------
User: Tell me about machine learning
Bot: Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions. Instead, these systems learn from data and improve their performance over time.

Key concepts in machine learning include:

1. **Types of Learning**:
- **Supervised Learning**: The model is trained on labeled data, meaning the input data is paired with the correct output. Common algorithms include linear regression, decision trees, and support vector machines.
- **Unsupervised Learning**: The model works with unlabeled data and tries to find patterns or groupings. Examples include clustering algorithms like k-means and hierarchical clustering.
- **Reinforcement Learning**: The model learns by interacting with an environment and receiving feedback in the form of rewards or penalties. This approach is often used in robotics and game playing.

2. **Common Algorithms**:
- **Linear Regression**: Used for predicting a continuous output based on input features.
- **Decision Trees**: A flowchart-like structure used for classification and regression tasks.
- **Neural Networks**: Inspired by the human brain, these are used for complex tasks like image and speech recognition.
- **Support Vector Machines (SVM

======================================================================
✓ All tests completed!

Expected Behavior:
- Test 1 & 3: Should provide grounded answers from knowledge base
- Test 2: Should return 'no relevant information' message



## Example 4: Source Attribution and Citation




Ensure all answers include proper source citations.


# Configuration with source attribution


citation_config = """

models:

- type: main

engine: openai

model: gpt-4o-mini

parameters:

temperature: 0.0

max_tokens: 256



instructions:

- type: general

content: |

You are a research assistant.

Always cite sources for information you provide.

Format citations as [Source: filename]



rails:

retrieval:

flows:

- retrieve with citations

output:

flows:

- verify citations

"""



# Colang for citations


citation_colang = """

define flow retrieve with citations

user ask question

$docs = execute retrieve_and_track_sources

bot provide cited answer



define subflow verify citations

$has_citations = execute check_citations



if not $has_citations

bot add citations



define bot add citations

"[Note: Please refer to source documents for verification]"

"""



# Citation actions


citation_actions = '''

from nemoguardrails.actions import action

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

import re



@action(is_system_action=True)

async def retrieve_and_track_sources(context: dict):

"""Retrieve documents and track sources for citation."""



user_message = context.get("last_user_message", "")



# Load vector store

embeddings = HuggingFaceEmbeddings(

model_name="sentence-transformers/all-MiniLM-L6-v2"

)

vectorstore = FAISS.load_local(

"configs/rag_basic/kb",

embeddings,

allow_dangerous_deserialization=True

)



# Retrieve documents

results = vectorstore.similarity_search_with_score(user_message, k=3)



# Build context with sources

context_parts = []

sources = []



for doc, score in results:

source = doc.metadata.get("source", "unknown")

context_parts.append(f"{doc.page_content} [Source: {source}]")

sources.append(source)



# Store in context

context["retrieval_context"] = "\n\n".join(context_parts)

context["sources"] = list(set(sources))  # Unique sources



return {

"context": context["retrieval_context"],

"sources": context["sources"]

}



@action(is_system_action=True)

async def check_citations(context: dict):

"""Check if bot response includes citations."""



bot_message = context.get("bot_message", "")



# Look for citation patterns

citation_patterns = [

r"\[Source: .+?\]",

r"\(Source: .+?\)",

r"according to .+",

r"from .+?\.txt"

]



has_citations = any(

re.search(pattern, bot_message, re.IGNORECASE)

for pattern in citation_patterns

)



return has_citations

'''



# Save configuration


!mkdir -p configs/citation_tracking



with open('configs/citation_tracking/config.yml', 'w') as f:

f.write(citation_config)



with open('configs/citation_tracking/config.co', 'w') as f:

f.write(citation_colang)



with open('configs/citation_tracking/actions.py', 'w') as f:

f.write(citation_actions)




# Copy vector store


!cp -r configs/rag_basic/kb configs/citation_tracking/



print("✓ Citation tracking configuration saved")


Output:
✓ Citation tracking configuration saved




# Test Example 4: Source Attribution and Citation


print("Testing Citation Tracking Configuration...")

print("=" * 70)




# Load the configuration


config_citation = RailsConfig.from_path("configs/citation_tracking")

rails_citation = LLMRails(config_citation)



print("✓ Citation tracking configuration loaded successfully\n")




# Test queries that should include citations


test_queries = [

"What is Python and who created it?",

"Explain what vector databases are used for",

"What is machine learning?"

]



for idx, query in enumerate(test_queries, 1):

print(f"Test {idx}: {query}")

print("-" * 70)



response = rails_citation.generate(messages=[{"role": "user", "content": query}])

print(f"Bot: {response['content']}")



# Check if citations are present

has_citation = '[Source:' in response['content'] or 'source' in response['content'].lower()

citation_status = "✓ Citations included" if has_citation else "⚠ No citations found"

print(f"\n{citation_status}\n")



print("=" * 70)

print("✓ Citation tracking tests completed!")

print("\nExpected: Responses should include [Source: filename] citations")


Output:
ERROR:nemoguardrails.actions.action_dispatcher:Failed to register actions.py in action dispatcher due to exception: unterminated string literal (detected at line 36) (actions.py, line 36)



Output:
Testing Citation Tracking Configuration...
======================================================================
✓ Citation tracking configuration loaded successfully

Test 1: What is Python and who created it?
----------------------------------------------------------------------
Bot: Action 'check_citations' not found.

⚠ No citations found

Test 2: Explain what vector databases are used for
----------------------------------------------------------------------
Bot: Action 'check_citations' not found.

⚠ No citations found

Test 3: What is machine learning?
----------------------------------------------------------------------
Bot: Action 'check_citations' not found.

⚠ No citations found

======================================================================
✓ Citation tracking tests completed!

Expected: Responses should include [Source: filename] citations



## Example 5: Access Control for Retrieval




Implement permission-based retrieval to control access to sensitive information.


# Create documents with different access levels


secured_documents = [

Document(

page_content="Public API documentation: Our REST API is available at api.example.com. Rate limit: 1000 requests/hour.",

metadata={"source": "public_api.txt", "access_level": "public"}

),

Document(

page_content="Internal server configuration: Production database host is db-prod-01.internal. Connection pool size: 50.",

metadata={"source": "internal_config.txt", "access_level": "internal"}

),

Document(

page_content="Confidential: Annual revenue for Q4 2024 is $10M. This information is not yet public.",

metadata={"source": "financials.txt", "access_level": "confidential"}

),

Document(

page_content="General company information: We were founded in 2020 and have 100+ employees.",

metadata={"source": "about.txt", "access_level": "public"}

)

]




# Create separate vector store


secured_vectorstore = FAISS.from_documents(secured_documents, embeddings)

!mkdir -p configs/access_control/kb

secured_vectorstore.save_local("configs/access_control/kb")



print("✓ Secured documents vector store created")


Output:
✓ Secured documents vector store created




# Configuration with access control


access_config = """

models:

- type: main

engine: openai

model: gpt-4o-mini

parameters:

temperature: 0.0

max_tokens: 256



instructions:

- type: general

content: |

You are a company information assistant.

Only provide information the user is authorized to access.



rails:

retrieval:

flows:

- check access and retrieve

"""



# Colang for access control


access_colang = """

define flow check access and retrieve

user ask question

$result = execute retrieve_with_access_control



if $result.access_denied

bot inform access denied

stop

else

bot provide authorized answer



define bot inform access denied

"You don't have permission to access that information. Please contact your administrator."

"""



# Access control actions


access_actions = '''

from nemoguardrails.actions import action

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings




# User access levels (in real app, would come from authentication)


USER_ACCESS_LEVELS = {

"public_user": ["public"],

"internal_user": ["public", "internal"],

"admin_user": ["public", "internal", "confidential"]

}



@action(is_system_action=True)

async def retrieve_with_access_control(context: dict):

"""Retrieve documents with access control."""



user_message = context.get("last_user_message", "")



# Get user access level (hardcoded for demo, would come from auth)

user_type = context.get("user_type", "public_user")

allowed_levels = USER_ACCESS_LEVELS.get(user_type, ["public"])



# Load vector store

embeddings = HuggingFaceEmbeddings(

model_name="sentence-transformers/all-MiniLM-L6-v2"

)

vectorstore = FAISS.load_local(

"configs/access_control/kb",

embeddings,

allow_dangerous_deserialization=True

)



# Retrieve documents

results = vectorstore.similarity_search_with_score(user_message, k=5)



# Filter by access level

authorized_docs = []

denied_docs = []



for doc, score in results:

access_level = doc.metadata.get("access_level", "public")



if access_level in allowed_levels:

authorized_docs.append({

"content": doc.page_content,

"source": doc.metadata.get("source", "unknown"),

"access_level": access_level

})

else:

denied_docs.append(access_level)



return {

"documents": authorized_docs,

"access_denied": len(denied_docs) > 0 and len(authorized_docs) == 0,

"authorized_count": len(authorized_docs),

"denied_count": len(denied_docs)

}

'''



# Save configuration


!mkdir -p configs/access_control



with open('configs/access_control/config.yml', 'w') as f:

f.write(access_config)



with open('configs/access_control/config.co', 'w') as f:

f.write(access_colang)



with open('configs/access_control/actions.py', 'w') as f:

f.write(access_actions)



print("✓ Access control configuration saved")


Output:
✓ Access control configuration saved




# Test Example 5: Access Control for Retrieval


print("Testing Access Control Configuration...")

print("=" * 70)




# Load the configuration


config_access = RailsConfig.from_path("configs/access_control")

rails_access = LLMRails(config_access)



print("✓ Access control configuration loaded successfully\n")




# Test with public user


print("Test as PUBLIC_USER")

print("=" * 70)

query1 = "Tell me about your API"

response1 = rails_access.generate(

messages=[{"role": "user", "content": query1}],

state={"user_type": "public_user"}

)

print(f"Query: {query1}")

print(f"Bot: {response1['content']}\n")




# Test confidential access as public user (should be denied)


query2 = "What is the company revenue?"

response2 = rails_access.generate(

messages=[{"role": "user", "content": query2}],

state={"user_type": "public_user"}

)

print(f"Query: {query2}")

print(f"Bot: {response2['content']}")

print("Expected: Access denied\n")




# Test as admin user (should have access)


print("Test as ADMIN_USER")

print("=" * 70)

response3 = rails_access.generate(

messages=[{"role": "user", "content": query2}],

state={"user_type": "admin_user"}

)

print(f"Query: {query2}")

print(f"Bot: {response3['content']}")

print("Expected: Access granted with information\n")



print("=" * 70)

print("✓ Access control tests completed!")


Output:
Testing Access Control Configuration...
======================================================================
✓ Access control configuration loaded successfully

Test as PUBLIC_USER
======================================================================



## Example 6: Hybrid Retrieval with Re-ranking




Combine multiple retrieval strategies and re-rank results.


# Hybrid retrieval action


hybrid_actions = '''

from nemoguardrails.actions import action

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import List, Dict



@action(is_system_action=True)

async def hybrid_retrieve_and_rerank(context: dict):

"""Hybrid retrieval with re-ranking."""



user_message = context.get("last_user_message", "")



# Load vector store

embeddings = HuggingFaceEmbeddings(

model_name="sentence-transformers/all-MiniLM-L6-v2"

)

vectorstore = FAISS.load_local(

"configs/rag_basic/kb",

embeddings,

allow_dangerous_deserialization=True

)



# Semantic search

semantic_results = vectorstore.similarity_search_with_score(user_message, k=10)



# Re-rank based on multiple factors

ranked_results = []

for doc, score in semantic_results:

# Calculate composite score

relevance_score = 1.0 / (1.0 + score)  # Normalize distance to 0-1



# Boost score based on metadata

category = doc.metadata.get("category", "")

category_boost = 1.2 if category in user_message.lower() else 1.0



final_score = relevance_score * category_boost



ranked_results.append({

"content": doc.page_content,

"source": doc.metadata.get("source", "unknown"),

"category": category,

"score": final_score

})



# Sort by final score

ranked_results.sort(key=lambda x: x["score"], reverse=True)



return {

"documents": ranked_results[:3],  # Top 3

"total_retrieved": len(ranked_results)

}

'''



print("Hybrid retrieval action defined")



# Test Example 6: Hybrid Retrieval with Re-ranking


print("Testing Hybrid Retrieval (Manual Test)...")

print("=" * 70)

print("\nNote: Example 6 demonstrates the hybrid_retrieve_and_rerank action.")

print("This action combines semantic search with re-ranking.\n")




# Manual test using the action directly


from nemoguardrails.actions import action

import sys

sys.path.append('configs/rag_basic')



print("How Hybrid Retrieval Works:")

print("-" * 70)

print("1. Semantic search retrieves top 10 candidates from vector store")

print("2. Each result gets a composite score:")

print("   - Base: Relevance score from similarity search")

print("   - Boost: 1.2x multiplier if category matches query")

print("3. Results are sorted by final score")

print("4. Top 3 results are returned")

print("5. LLM generates answer from re-ranked documents\n")



print("Example Benefits:")

print("-" * 70)

print("✓ Better ranking than pure semantic search")

print("✓ Category-aware retrieval")

print("✓ Configurable scoring weights")

print("✓ Flexible re-ranking strategies\n")



print("To use hybrid retrieval in production:")

print("-" * 70)

print("1. Create config with hybrid_retrieve_and_rerank action")

print("2. Define Colang flow that calls the action")

print("3. Customize scoring logic in the action")

print("4. Test with various query types\n")



print("=" * 70)

print("✓ Hybrid retrieval concept explained!")

print("\nFor full implementation, create a config using the action from cell above.")


## Summary




In this notebook, we covered:



1. **Knowledge Base Integration**: Setting up vector stores with NeMo Guardrails

2. **Relevance Filtering**: Filtering retrieved content by relevance scores

3. **Source Attribution**: Tracking and citing information sources

4. **Access Control**: Implementing permission-based retrieval

5. **Quality Scoring**: Ranking and filtering retrieval results

6. **Hybrid Retrieval**: Combining multiple strategies with re-ranking




## Key Takeaways




Retrieval rails ensure RAG systems return appropriate information

Relevance filtering prevents irrelevant information from reaching the LLM

Source attribution builds trust and enables verification

Access control prevents unauthorized information disclosure

Re-ranking improves result quality




## Next Steps


**Notebook 05**: Dialog Rails and Custom Actions