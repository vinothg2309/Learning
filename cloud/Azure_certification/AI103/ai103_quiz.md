# Microsoft AI & Agentic Solutions Exam Prep

## Speech processing solution
**Question:** You are building a speech processing solution in Microsoft Foundry to transcribe live phone calls so supervisors can view transcripts and detect issues in real-time. What should you do?

* [ ] Use text-to-speech
* [ ] Use speech translation
* [ ] Use batch transcription
* [x] Use real-time speech-to-text to process streaming audio input

**Reasoning:** The requirement for "live" calls and transcripts appearing within "a few seconds" necessitates low-latency, continuous streaming processing rather than post-call batch processing (3:07).

---

## Extracting content from scanned invoices
**Question:** You have a Microsoft Foundry project with an agent that ingests scanned PDF vendor invoices containing tables and QR codes. You need to extract content and layout elements and detect QR codes without requiring a language model deployment. Which built-in analyzer should you use?

* [ ] Pre-built document field schema
* [ ] Pre-built read
* [ ] Pre-built document search
* [x] Pre-built layout

**Reasoning:** The pre-built layout analyzer extracts structure (tables, paragraphs) and detects QR codes while preserving layout, without needing a language model (5:17).

---

## Configuring prompt shields
**Question:** You have an agent that accepts user-uploaded screenshots. Some contain malicious embedded text. You need to prevent prompt injection attacks and ensure third-party content is treated as lower trust. How should you configure prompt shields?

* [ ] Set action to annotate and disable shield
* [x] Set action to block and enable spotlighting
* [ ] Set action to block and create a custom block list
* [ ] Use OCR first

**Reasoning:** "Block" stops the injection attack, and "Spotlighting" helps the model distinguish between system instructions and untrusted third-party content (7:41).

---

## Integrating applications with Azure AI Search
**Question:** You are building a web app that uses a model in Microsoft Foundry. Before sending prompts, the app must retrieve documents using Azure AI Search. Multiple applications must use the same search configuration, a security policy prevents key-based authentication, and administrative effort must be minimized. What should you do?

* [ ] Manually configure endpoints for each app
* [x] Configure an Azure AI search connection in the Microsoft Foundry project and reference it in each application
* [ ] Use Entra ID for each app individually
* [ ] Use managed identities for each app

**Reasoning:** Centralizing the connection in the project allows reuse across applications, simplifies management, and supports Entra ID (token-based) authentication (10:25).

---

## Inspecting agent run observability
**Question:** Users report that some requests take more than 15 seconds or return incorrect responses. You need to inspect individual agent runs to view the ordered sequence of LLM calls, tool invocations, and timing information. Which observability capability should you use?

* [ ] Token usage
* [ ] Monitoring
* [ ] Safety metrics
* [x] Tracing

**Reasoning:** Tracing records the sequential execution flow, tool usage, and timing for specific agent runs, allowing for granular troubleshooting (12:51).

---

## Optimizing agent workflow tools
**Question:** You need to optimize an agent's workflow by providing capabilities to: access up-to-date information from public websites, perform calculations during conversations, and retrieve information from uploaded documents. Which tools should you use for each?

* [ ] Grounding with Bing search
* [ ] Code interpreter
* [ ] File search
* [x] Access public websites: Grounding with Bing search; Perform calculations: Code interpreter; Retrieve from documents: File search

**Reasoning:** These specific tools are designed for real-time web access, computational code execution, and indexed retrieval from internal files, respectively (15:20).

---

## API key configuration in Open API specs
**Question:** An agent uses an Open API 3.0 specification to call an external weather service. The service requires a key passed in an HTTP header, and the key is stored as a connection in the project. How do you ensure the key is included automatically?

* [ ] Define a header parameter
* [ ] Use Azure Key Vault
* [x] Configure an API key security scheme
* [ ] Use a bearer token security scheme

**Reasoning:** An API key security scheme in the spec tells the tool to automatically inject the key from the configured project connection into the header (17:46).

---

## Orchestrating multiple agents
**Question:** You have three agents (Triage, Policy, and Action). You need to orchestrate them to support a deterministic step-based process that uses conditional branching and shared state while minimizing development effort. What should you include?

* [x] A workflow
* [ ] Threads and runs
* [ ] Multi-agent group chat
* [ ] Application code orchestration

**Reasoning:** Workflows provide a built-in, deterministic way to handle branching logic and shared state across agent steps without writing extensive custom code (20:11).

---

## GitHub Actions quality gates
**Question:** An agent is deployed using GitHub Actions. You must ensure every deployment uses the latest approved evaluation baseline and that deployments stop automatically if evaluation scores regress beyond the tolerance. How do you configure the workflow?

* [ ] Compare against previous run/retry
* [x] Compare against the latest approved baseline; if regression exceeds tolerance, fail the workflow
* [ ] Compare against repo default branch/lock branch

**Reasoning:** Comparing against the approved baseline enforces quality standards, and failing the workflow prevents the deployment of regressed agents (22:14).

---

## Telemetry configuration
**Question:** For a pilot project, you have specific company policies regarding telemetry distinction and sensitive data. Are these statements true or false:
1. LangChain appears in traces without configuring a tracer.
2. Different service names separate telemetry.
3. Setting 'enable content recording' to false still captures prompts in telemetry.

* [ ] 1. Yes, 2. Yes, 3. Yes
* [ ] 1. Yes, 2. No, 3. No
* [x] 1. No, 2. Yes, 3. No
* [ ] 1. No, 2. No, 3. Yes

**Reasoning:** LangChain needs an explicit tracer; unique service names provide logical separation in Application Insights; and disabling content recording prevents the capture of prompts/arguments (25:41).

---
1. Invoice Processing Analyzer (0:01 - 2:23)
You have a Microsoft Foundry project that ingests scanned PDF invoices stored in Azure Blob storage. Each invoice contains printed line items and has a table-based layout. Extracted results are stored as structured JSON and used as grounding data for an agent in a RAG solution. You need to create a single analyzer that extracts the invoice number, invoice date, vendor name, and total amount across varying templates and returns confidence scores so that results with confidence below 0.80 can be routed for supervisor review. What should you use?

[ ] Groundedness guardrails
[x] A custom Azure content understanding analyzer in Foundry tools
[ ] A pre-built layout analyzer
[ ] Search score Reasoning: A custom analyzer (0:48) allows definition of specific fields and provides extraction confidence scores, unlike layout analyzers (which focus on structure) or groundedness guardrails (which focus on response verification).
---
2. Indirect Prompt Injection (2:26 - 4:18)
A Microsoft Foundry agent retrieves content from an external website by using grounding with Bing search. Security administrators want to detect indirect prompt injection attempts and review detections before deciding whether to block requests. Which configuration should you recommend?

[ ] Disable prompt shields
[ ] Configure prompt shields to block
[x] Configure prompt shields to annotate and enable spotlighting
[ ] Enable content safety Reasoning: Annotate mode (3:10) flags attempts without automatically blocking them, and spotlighting (3:30) helps the model distinguish trusted system instructions from malicious external content.
---
3. Power Fx Expressions (4:19 - 6:37)
You have a Microsoft Foundry project that contains a workflow for a customer support triage process. You have an "Ask a question" node that stores user responses in a local variable named var01. You need to create an if-else condition expression that ensures var01 contains a value, and a send message expression that returns the stored user response in uppercase. How should you configure these?

[x] !IsBlank(local.var01) and Upper(local.var01)
[ ] IsBlank(local.var01) and local.var01
[ ] IsEmpty(local.var01) and local.var01
[ ] Upper(101) and !IsBlank(101) Reasoning: !IsBlank (5:24) checks if the variable has data, and the Upper function (5:49) converts the string to uppercase.
4. Multi-turn Continuity (6:38 - 8:43)
You have a customer support agent that uses the Microsoft Foundry Agent Service. Sometimes customers return to a session days later to continue the same support case, and the agent must resume with the full historical context. You need to ensure the agent provides multi-turn and cross-session continuity, including access to user messages, agent messages, tool calls, and tool outputs. What should you do?

[x] Create and reuse a conversation by storing the conversation's ID and supplying the ID on subsequent requests
[ ] Store only the final model response
[ ] Use memory summarization Reasoning: Supplying the same conversation ID (7:50) allows the Agent Service to automatically reload the full interaction history.
5. HR Onboarding Thread Management (8:44 - 10:47)
You have a Microsoft Foundry Agent Service project that contains an HR onboarding agent. Employees frequently pause conversations and continue them several days later. The solution must resume the conversation exactly where it ended, ensure the agent can reference previous tool invocations and uploaded documents, and avoid manually reconstructing history. What should you do?

[ ] Create a new thread for every request
[x] Persist the thread ID and continue subsequent requests by using the existing thread
[ ] Use conversation summaries
[ ] Store only the final assistant response Reasoning: Persisting the thread ID (9:48) allows the agent to reload the full context, including tool invocations and uploaded files, without manual reconstruction.
6. Regulatory Compliance Pipelines (13:47 - 16:08)
You have a Microsoft Foundry project that processes regulatory compliance documents submitted by regional offices. You need to implement two pipelines: Pipeline 1 extracts key compliance information from individual audit reports while minimizing processing costs; Pipeline 2 analyzes multiple related audit reports together to identify inconsistencies across reporting periods by using advanced reasoning. How should you configure each pipeline?

[x] Pipeline 1: Single file task in standard mode; Pipeline 2: Multifile task in pro mode
[ ] Pipeline 1: Multifile task in standard mode; Pipeline 2: Single file task in pro mode
[ ] Pipeline 1: Single file task in pro mode; Pipeline 2: Multifile task in standard mode Reasoning: Single file/standard mode (15:17) is cost-effective for isolated documents, while pro mode with multifile capabilities (15:41) is required for cross-document reasoning.
7. Operational Cost Observability (16:09 - 18:07)
You have a Microsoft Foundry project that contains a high-traffic agent. After a recent update, operational costs increased significantly, even though user traffic volume remains unchanged. You need to identify whether the additional costs are driven by model input size, model output size, or expanded tool usage. Which observability capability should you use?

[ ] Latency
[ ] Evaluation metrics
[ ] Run success rate
[x] Token usage Reasoning: Token usage (17:11) provides granular breakdown of inputs, outputs, and tool consumption, making it the correct tool for investigating cost drivers.
8. RAG Agent RBAC (18:08 - 19:48)
You have a Microsoft Foundry project that contains an Azure storage account used to store documents for RAG. The agents authenticate by using managed identities. The agents must be able to read blob data but must not upload, modify, or delete files. Which RBAC role should you assign to the managed identities?

[ ] Storage blob data owner
[ ] Storage blob data contributor
[x] Storage blob data reader
[ ] Contributor Reasoning: The Storage Blob Data Reader role (18:54) grants read-only access, adhering to the principle of least privilege.
9. CI/CD Evaluation (19:49 - 21:55)
You have a Microsoft Foundry project that contains an agent. You use a GitHub Actions workflow for CI/CD. You need to configure the workflow to automatically evaluate the agent when a pull request is created and prevent branches from merging if the evaluation results do not meet the defined thresholds. How should you configure the workflow?

[x] Azure login action that uses OpenID Connect and set the action to "fail" if thresholds aren't met
[ ] Personal access token and "lock" the target branch
[ ] User-assigned managed identity and "send an alert" Reasoning: OpenID Connect (20:45) is the secure method for Azure authentication, and "failing" the workflow (21:07) blocks the merge in GitHub.
10. Custom Speech Model Integration (21:56 - 23:38)
You have a Microsoft Foundry project that contains an agent using Azure Speech. You fine-tuned a baseline speech-to-text model for the en-US locale and published it. The agent calls a speech-to-text REST API and returns an error message indicating that the project ID is invalid. To what should you set the project property?

[ ] The project URL
[x] The custom speech project ID
[ ] The Microsoft Foundry project ID
[ ] The custom speech endpoint URL Reasoning: When using a custom speech model, the API requires the specific custom speech project ID (22:53) to locate the published model.