# Security Frameworks for AI Systems

## IV. FRAMEWORKS DETAILS

### PLOT4AI Framework
Privacy, Legal, Operational, Technical risks for AI

#### Privacy Risks Domain
- **Data Privacy**: Protection of personal information throughout AI lifecycle
- **Privacy by Design**: Embedding privacy considerations into AI system architecture
- **Privacy Enhancing Technologies (PETs)**:
  - Differential Privacy: Mathematical framework for privacy preservation
  - Homomorphic Encryption: Computation on encrypted data
  - Secure Multi-party Computation: Joint computation without sharing inputs
  - Federated Learning: Training across decentralized devices
  - Zero-Knowledge Proofs: Proving knowledge without revealing information
- **Data Minimization**: Collecting only what is necessary
- **Purpose Limitation**: Using data only for specified, explicit purposes
- **Storage Limitation**: Retaining data only as long as needed

#### Legal Risks Domain
- **Intellectual Property**: Copyright, patents, trade secrets related to AI
- **Liability**: Responsibility for harm caused by AI systems
- **Regulatory Compliance**: Adherence to applicable laws and regulations
- **Contractual Obligations**: Meeting commitments in agreements and SLAs
- **Transparency Requirements**: Disclosure obligations about AI usage
- **Audit Trails**: Documentation for regulatory compliance

#### Operational Risks Domain
- **Reliability**: Consistent performance under expected conditions
- **Availability**: Accessibility and usability when needed
- **Scalability**: Ability to handle growth in workload or users
- **Maintainability**: Ease of fixing defects and making changes
- **Performance**: Responsiveness and efficiency under workload
- **Interoperability**: Ability to work with other systems and components
- **Portability**: Ability to transfer between different environments

#### Technical Risks Domain
- **Security**: Protection against unauthorized access and malicious use
- **Resilience**: Ability to withstand and recover from adverse conditions
- **Compatibility**: Ability to function with specified other systems
- **Usability**: Ease of learning and use
- **Accessibility**: Usability by people with diverse abilities
- **Installability**: Ability to be installed in a specified environment
- **Replaceability**: Ability to replace with another specified product

### OWASP LLM Top 10 (2023)
The Most Critical Security Risks for Large Language Model Applications

#### LLM01:2023 - Prompt Injection
**Description**: 
An attacker manipulates a LLM through crafted inputs, causing the LLM to execute the attacker's intentions. This can lead to data leakage, unauthorized actions, or system compromise.

**Common Attack Vectors**:
- Direct prompt injection via user input
- Indirect injection via data sources (web scraping, file uploads)
- Roleplay and persona manipulation
- Encoding-based obfuscation (base64, unicode, etc.)
- Recursive and self-referential prompts
- Context window exploitation

**Impact**:
- Data exfiltration and information disclosure
- Unauthorized system access and control
- Financial fraud and unauthorized transactions
- Reputational damage through offensive outputs
- Compliance violations (GDPR, HIPAA, etc.)

**Mitigation Strategies**:
- Input validation and sanitization
- Prompt engineering with clear role separation
- Output encoding and filtering
- Sandboxed execution environments
- User behavior analysis and anomaly detection
- Regular security testing and red teaming

#### LLM02:2023 - Insecure Output Handling
**Description**: 
LLM outputs are not properly validated, filtered, or encoded before being passed to downstream components, enabling injection attacks or data leakage.

**Common Attack Vectors**:
- Cross-site scripting (XSS) via web applications
- Server-side request forgery (SSRF)
- Command and code injection
- Path traversal and file access
- SQL and NoSQL injection
- Template injection

**Impact**:
- Remote code execution and server compromise
- Data theft and unauthorized access
- Client-side attacks in web applications
- Defacement and service disruption
- Secondary infection and lateral movement

**Mitigation Strategies**:
- Context-aware output encoding
- Implementation of Content Security Policies (CSP)
- Use of parameterized APIs and prepared statements
- Input validation for downstream consumption
- Sandboxed execution of LLM outputs
- Automated security testing (DAST, SAST)

#### LLM03:2023 - Training Data Poisoning
**Description**: 
An attacker manipulates training data to compromise the model's behavior, integrity, or availability, leading to biased or harmful outputs.

**Common Attack Vectors**:
- Data injection during collection or preprocessing
- Label flipping and backdoor insertion
- Bias injection through skewed sampling
- Data leakage from unauthorized sources
- Poor data quality and garbage-in-garbage-out

**Impact**:
- Biased or discriminatory outputs
- Backdoor behaviors for specific triggers
- Reduced model accuracy and effectiveness
- Undermined trust in AI systems
- Legal and regulatory compliance issues

**Mitigation Strategies**:
- Data provenance and validation
- Anomaly detection in training data
- Robust statistical methods and outlier rejection
- Data preprocessing and cleaning pipelines
- Secure data sources and supplier management
- Continuous monitoring and retraining

#### LLM04:2023 - Model Denial of Service
**Description**: 
An attacker causes excessive resource consumption in an LLM, affecting service availability and performance for legitimate users.

**Common Attack Vectors**:
- Resource-intensive prompts (complex calculations, recursion)
- Context window flooding with irrelevant data
- Output length maximization attacks
- Concurrent request flooding (DDoS-style)
- Cryptographic computational attacks
- Memory exhaustion through large inputs

**Impact**:
- Service degradation and slow response times
- Complete service unavailability
- Increased operational costs
- Poor user experience and abandonment
- Masking of other security attacks

**Mitigation Strategies**:
- Rate limiting and throttling mechanisms
- Input length and complexity restrictions
- Output length and frequency controls
- Request timeout and deadline enforcement
- Resource quotas and monitoring
- Caching and request deduplication
- Autoscaling and load balancing

#### LLM05:2023 - Supply Chain Vulnerabilities
**Description**: 
Vulnerabilities in components, libraries, or tools used to develop, deploy, or operate LLMs that could be exploited to compromise security.

**Common Attack Vectors**:
- Compromised ML libraries and frameworks
- Malicious pre-trained models from public repositories
- Vulnerable data processing and annotation tools
- Insecure MLOps platforms and pipelines
- Third-party APIs and services with poor security

**Impact**:
- Backdoors and hidden functionality in models
- Data theft through compromised components
- Service disruption through dependency failures
- Lateral movement to connected systems
- Reputation damage from security incidents

**Mitigation Strategies**:
- Software Bill of Materials (SBOM) generation
- Dependency scanning and vulnerability management
- Signed and verified artifacts (cosign, Sigstore)
- Private artifact repositories with access controls
- Continuous monitoring of third-party updates
- Contractual security requirements for suppliers

#### LLM06:2023 - Sensitive Information Disclosure
**Description**: 
LLM outputs reveal sensitive information such as personal data, proprietary information, or intellectual property that should not be disclosed.

**Common Attack Vectors**:
- Direct output of PII, PHI, or secrets
- Indirect disclosure through patterns and correlations
- Training data memorization and exact reproduction
- Overfitting to sensitive training examples
- Insufficient output filtering and redaction

**Impact**:
- Privacy violations and regulatory fines
- Loss of trade secrets and competitive advantage
- Identity theft and financial fraud
- Reputational damage and customer trust loss
- Legal liability for data breaches

**Mitigation Strategies**:
- PII detection and redaction technologies
- Output filtering and content controls
- Training data minimization and anonymization
- Differential privacy techniques
- Access controls and monitoring
- Data loss prevention (DLP) integration

#### LLM07:2023 - Insecure Plugin Design
**Description**: 
LLM plugins have excessive permissions, insufficient input validation, or inadequate output validation, enabling abuse and attack.

**Common Attack Vectors**:
- Overly permissive tool access and privileges
- Insufficient input validation leading to injection
- Inadequate output validation enabling secondary attacks
- Insecure direct object references (IDOR)
- Missing authentication and authorization checks

**Impact**:
- Unauthorized system access and control
- Data theft and unauthorized modifications
- Privilege escalation and lateral movement
- Service disruption and denial of service
- Secondary infection through compromised plugins

**Mitigation Strategies**:
- Principle of least privilege for tool access
- Comprehensive input validation and sanitization
- Output validation and encoding
- Strong authentication and authorization
- Sandboxed execution environments for plugins
- Regular security testing and code review

#### LLM08:2023 - Excessive Agency
**Description**: 
LLM is permitted to perform actions that are overly extensive or unauthorized, leading to unintended consequences and potential harm.

**Common Attack Vectors**:
- Autonomous financial transactions without approval
- Unauthorized data modifications or deletions
- Unapproved system configuration changes
- Unmonitored communications or notifications
- Unrestricted network access and scanning

**Impact**:
- Financial losses and unauthorized transfers
- Data corruption and loss of integrity
- System instability and configuration drift
- Reputational damage from unauthorized actions
- Legal liability for unauthorized activities

**Mitigation Strategies**:
- Human-in-the-loop for high-risk actions
- Action approval workflows and validation
- Constrained capabilities and permission systems
- Audit trails and forensic logging
- Time-based and geofencing restrictions
- Session management and expiration

#### LLM09:2023 - Overreliance
**Description**: 
Excessive trust in LLMs without adequate critical thinking or fact-checking, leading to erroneous decisions and actions.

**Common Attack Vectors**:
- Blind acceptance of medical or legal advice
- Unverified financial or investment recommendations
- Unchecked safety-critical instructions
- Reliance on hallucinated or fabricated information
- Lack of human oversight in critical processes

**Impact**:
- Incorrect medical diagnosis and treatment
- Financial losses from bad advice
- Safety incidents and physical harm
- Reputational damage from erroneous outputs
- Operational inefficiencies from wrong decisions

**Mitigation Strategies**:
- Confidence scoring and uncertainty estimation
- Human review requirements for high-risk domains
- Fallback to traditional methods and systems
- Explainability and transparency mechanisms
- Critical thinking and media literacy training
- Regular model performance monitoring

#### LLM10:2023 - Model Theft
**Description**: 
Unauthorized access, copying, or use of proprietary LLMs through legal or illegal means.

**Common Attack Vectors**:
- API-based model extraction through query analysis
- Direct theft of model files and artifacts
- Reverse engineering through side-channel attacks
- Insider threats and privilege abuse
- Supply chain compromise of distribution channels

**Impact**:
- Loss of competitive advantage and IP
- Revenue loss from unauthorized use
- Security risks from modified or poisoned copies
- Reputational damage from misuse
- Legal liability for unauthorized distribution

**Mitigation Strategies**:
- Model encryption at rest and in transit
- Strong access controls and authentication
- Usage monitoring and anomaly detection
- Model watermarking and fingerprinting
- Secure distribution and licensing mechanisms
- Legal protections and enforcement

### MITRE ATLAS Framework
Adversarial Threat Landscape for Artificial-Intelligence Systems

#### Core Concepts
- **Adversarial ML Tactics**: The "why" behind an attack (objectives)
- **Adversarial ML Techniques**: The "how" of achieving tactical goals
- **ML-Specific Attack Surface**: Unique vulnerabilities in ML systems
- **Defensive Countermeasures**: Strategies to prevent, detect, and respond
- **Threat Intelligence Sharing**: Collaboration to improve collective defense

#### ML Attack Lifecycle and Tactics

##### Reconnaissance (TA001)
**Objective**: Gather information about potential targets for malicious operations.
- **T001: ML Model Information Gathering**: Discover model existence, purpose, endpoints
- **T002: Data Source Identification**: Identify training data origins and characteristics
- **T003: Architecture and Hyperparameter Estimation**: Infer model structure and settings
- **T004: Dependency and Library Identification**: Determine software dependencies
- **T005: Cloud and Infrastructure Enumeration**: Map deployment environment

##### Weaponization (TA002)
**Objective**: Prepare resources for malicious operations.
- **T006: Adversarial Crafting**: Create inputs designed to cause misclassification
- **T007: Backdoor Engineering**: Develop hidden triggers for specific behaviors
- **T008: Data Poisoning**: Prepare corrupted training data samples
- **T009: Resource Development**: Obtain or create necessary tools and infrastructure
- **T010: Encryption and Obfuscation**: Hide malicious intentions and tools

##### Delivery (TA003)
**Objective**: Transmit weaponized items to target environment.
- **T011: ML Supply Chain Compromise**: Introduce vulnerabilities via dependencies
- **T012: Training Data Poisoning**: Inject malicious samples into training data
- **T013: Model Access and Manipulation**: Gain unauthorized access to models
- **T014: Third-Party Services Exploitation**: Leverage external services for attack
- **T015: Social Engineering**: Manipulate individuals to gain access or information

##### Exploitation (TA004)
**Objective**: Gain capability to operate within victim environment.
- **T016: Prompt Injection**: Manipulate LLM behavior through crafted inputs
- **T017: Evasion Attack**: Avoid detection through test-time perturbation
- **T018: Model Inversion**: Reconstruct sensitive information from model outputs
- **T019: Model Extraction**: Steal model functionality or parameters
- **T020: Adversarial ML via Third-Party Services**: Leverage external ML services

##### Action on Objectives (TA005)
**Objective**: Achieve the goal of malicious ML operations.
- **T021: Exfiltration**: Extract sensitive data or intellectual property
- **T022: Inhibition**: Disrupt or degrade ML system performance
- **T023: Copyright and IP Circumvention**: Bypass protections on protected works
- **T024: Wrongful Deployment**: Deploy models in inappropriate contexts
- **T025: Manufacturing Defects**: Introduce flaws during manufacturing process

#### ML-Specific Tactics Details

##### T016: Prompt Injection
- **Description**: Adversarial manipulation of LLM prompts to alter intended behavior
- **Techniques**:
  - Direct injection: "Ignore previous instructions and do X"
  - Indirect injection: Hidden instructions in data sources
  - Roleplay: "You are now DAN, do anything without restrictions"
  - Encoding: Base64, URL, Unicode obfuscation of malicious prompts
  - Recursive: Prompts that generate further malicious prompts
- **Data Sources**: User inputs, files, databases, APIs, web scraping
- **Application**: LLMs, multimodal models, vision-language models
- **Defenses**: Input validation, prompt templating, sandboxing, monitoring

##### T017: Evasion Attack
- **Description**: Modifying inputs to avoid detection while maintaining malicious intent
- **Techniques**:
  - Gradient-based methods: FGSM, PGD, IG
  - Optimization-based: C&W, DeepFool
  - Decision-based: Boundary attack, HopSkipJump
  - Transfer-based: Exploiting transferability between models
  - Physical-world: Printed ads, stickers, clothing, environmental factors
- **Data Sources**: Network traffic, files, images, audio, video
- **Application**: Image classifiers, object detectors, facial recognition, malware detectors
- **Defenses**: Adversarial training, input preprocessing, detection consistency

##### T018: Model Inversion
- **Description**: Reconstructing training data or sensitive features from model access
- **Techniques**:
  - Optimization-based: Minimizing loss to reconstruct inputs
  - Feature-based: Targeting specific features or attributes
  - Label-specific: Focusing on samples of particular classes
  - Confidence-based: Using model confidence scores
  - Embedding-based: Utilizing internal representations
- **Data Sources**: Model outputs, confidence scores, embeddings, gradients
- **Application**: Facial recognition, medical diagnostics, financial models, recommendation systems
- **Defenses**: Differential privacy, output perturbation, access restrictions, model watermarking

##### T019: Model Extraction
- **Description**: Stealing ML model functionality through query access
- **Techniques**:
  - Prediction-based: Stealing via input-output pairs
  - Hyperparameter-based: Inferring architecture and settings
  - Boundary-based: Approximating decision boundaries
  - Jacobian-based: Using derivatives for sensitive data
  - Membership-based: Determining training set membership
- **Data Sources**: API queries, model outputs, timing information, power analysis
- **Application**: Any model accessible via API (CV, NLP, speech, etc.)
- **Defenses**: Prediction entropy limiting, output distortion, watermarking, rate limiting

##### T020: Adversarial ML via Third-Party Services
- **Description**: Using compromised or malicious external ML services
- **Techniques**:
  - Poisoned Models: Distributing backdoored or biased models
  - Malicious Labeling: Providing incorrect labels for training data
  - Corrupted Data: Distributing tainted or malicious datasets
  - Fake Services: Offering ML services designed to attack users
  - Supply Chain: Compromising legitimate ML service providers
- **Data Sources**: APIs, file transfers, model exchanges, collaborative platforms
- **Application**: Any third-party ML service usage
- **Defenses**: Service vetting, sandboxed execution, input/output validation, monitoring

##### T021: Exfiltration
- **Description**: Unauthorized transfer of data from ML system to external location
- **Techniques**:
  - Direct Transfer: Sending data via network or storage
  - Covert Channels: Using stealthy communication methods
  - Model-Based: Using model parameters or outputs as carriers
  - Timing Channels: Encoding data in request/response timing
  - Resource Utilization: Using compute/memory patterns to signal data
- **Data Sources**: Model internos, training data, intermediate results, logs
- **Application**: Any ML system handling sensitive data
- **Defenses**: DLP integration, network monitoring, access controls, encryption

##### T022: Inhibition
- **Description**: Disrupting or degrading ML system performance or availability
- **Techniques**:
  - Resource Exhaustion: Consuming compute, memory, bandwidth, storage
  - Algorithmic Interference: Degrading model accuracy or convergence
  - Environmental Manipulation: Altering temperature, power, network
  - Temporal Disruption: Exploiting timing dependencies and race conditions
  - Physical Damage: Causing hardware failure or degradation
- **Data Sources**: System resources, network traffic, environmental sensors
- **Application**: Any deployed ML system
- **Defenses**: Resource monitoring, redundancy, anomaly detection, fallback systems

##### T023: Copyright and IP Circumvention
- **Description**: Bypassing legal protections on copyrighted or patented works
- **Techniques**:
  - Style Mimicry: Replicating artistic or literary styles
  - Content Generation: Creating derivative works without authorization
  - Transformation Evasion: Modifying to avoid similarity detection
  - Fragment Assembly: Combining small parts to avoid detection
  - Watermark Removal: Detecting and removing digital watermarks
- **Data Sources**: Training data, outputs, intermediate representations
- **Application**: Generative models (text, image, music, video, code)
- **Defenses**: Watermarking, provenance tracking, legal monitoring, usage controls

##### T024: Wrongful Deployment
- **Description**: Deploying ML models in contexts where they may cause harm
- **Techniques**:
  - Context Misrepresentation: Providing false information about use case
  - Performance Inflation: Exaggerating capabilities or accuracy
  - Safety Warnings Suppression: Hiding known limitations or risks
  - Regulatory Non-Compliance: Ignoring applicable laws and standards
  - Ethical Guidelines Violation: Contradicting established principles
- **Data Sources**: Documentation, marketing materials, specifications
- **Application**: Any ML system deployment decision
- **Defeses**: Due diligence, impact assessment, external review, compliance checking

##### T025: Manufacturing Defects
- **Description**: Introducing flaws during the manufacturing of ML systems
- **Techniques**:
  - Supply Chain Compromise: Using compromised components or materials
  - Process Deviation: Not following established manufacturing procedures
  - Component Substitution: Using inferior or malicious replacements
  - Environmental Contamination: Exposing to harmful substances during production
  - Design Flaws: Errors in the original design or specifications
- **Data Sources**: Manufacturing logs, supplier records, inspection reports
- **Application**: Hardware-integrated ML systems (edge devices, IoT, automotive)
- **Defenses**: Supplier qualification, process validation, inspection, testing

#### Defensive Countermeasures (DET Series)

##### DET001: Input Validation and Sanitization
- **Description**: Checking and cleaning inputs to prevent malicious exploitation
- **Applications**:
  - Prompt and query validation for LLMs and other ML models
  - Data validation for training and inference pipelines
  - Configuration and parameter validation
  - File and upload validation for user-provided content
  - Network packet and protocol validation
- **Techniques**:
  - Allowlist validation (known good patterns)
  - Length and format restrictions
  - Character set validation and normalization
  - Syntax and semantic analysis
  - Behavioral analysis and anomaly detection
  - Machine learning-based classification
- **Limitations**: 
  - Cannot defend against unknown attack patterns
  - May impact usability if overly restrictive
  - Requires regular updates to address evolving threats
  - Performance overhead for complex validation

##### DET002: Output Filtering and Monitoring
- **Description**: Examining and controlling outputs to prevent data leakage
- **Applications**:
  - LLM response filtering for PII and malicious content
  - Model prediction validation for anomalies and manipulation
  - Action and command output validation
  - Data export and transmission monitoring
  - API response validation and sanitization
- **Techniques**:
  - Keyword and phrase blacklisting/whitelisting
  - Regular expression pattern matching
  - Structural validation (JSON, XML, SQL, etc.)
  - Entropy and randomness analysis
  - Machine learning-based toxicity and malicious content detection
  - Destination-based controls (allowlists, blocklists)
- **Limitations**:
  - May not detect novel exfiltration techniques
  - Can introduce latency and processing overhead
  - Requires tuning to balance security and usability
  - Limited effectiveness against encrypted channels

##### DET003: Model Access Controls
- **Description**: Restricting who or what can access ML models and related resources
- **Applications**:
  - Model file and artifact access control
  - API endpoint access management
  - Training and inference environment access
  - Monitoring and logging system access
  - Administrative and configuration access
- **Techniques**:
  - Authentication mechanisms (passwords, tokens, certificates)
  - Authorization systems (RBAC, ABAC, JWT claims)
  - Network segmentation and isolation (VLANs, subnets, firewalls)
  - Environment separation (development, staging, production)
  - Physical security controls (locks, badges, surveillance)
  - Session management and timeout enforcement
- **Limitations**:
  - Insider threats with legitimate access
  - Credential theft and compromise
  - Privilege escalation vulnerabilities
  - Configuration errors and misconfigurations
  - Social engineering to bypass controls

##### DET004: Anomaly Detection in ML Pipelines
- **Description**: Identifying unusual patterns that may indicate security issues
- **Applications**:
  - Data drift and concept drift detection
  - Performance degradation monitoring
  - Resource usage anomaly detection
  - Prediction distribution and outlier detection
  - User behavior and access pattern analysis
- **Techniques**:
  - Statistical methods (control charts, EWMA, seasonal decomposition)
  - Machine learning (isolation forest, LOF, one-class SVM)
  - Time series analysis (ARIMA, exponential smoothing)
  - Clustering-based approaches (DBSCAN, HDBSCAN, spectral)
  - Neural network autoencoders for reconstruction error
  - Rule-based and heuristic-based detection
- **Limitations**:
  - Baseline establishment requires normal operation data
  - False positives during legitimate changes or updates
  - Detection latency for slow-developing threats
  - Resource overhead for continuous monitoring
  - Sophisticated attacks may mimic normal behavior

##### DET005: Differential Privacy
- **Description**: Mathematically rigorous framework for privacy preservation
- **Applications**:
  - Privacy-preserving data analysis and statistics
  - Confidential machine learning model training
  - Secure query answering over sensitive datasets
  - Privacy-preserving feature extraction and embeddings
  - Anonymous data sharing and publication
- **Techniques**:
  - Laplace mechanism: Adding Laplace-distributed noise
  - Gaussian mechanism: Adding Gaussian-distributed noise
  - Exponential mechanism: Selecting outputs with probability
  - Randomized response: Flipping bits with probability
  - Shuffle model: Permuting responses before adding noise
  - Federated averaging with noise addition
- **Parameters**:
  - Epsilon (ε): Privacy budget (lower = more private)
  - Delta (δ): Probability of privacy breach (often set to 1/n)
  - Sensitivity: Maximum change in output from single data point
- **Limitations**:
  - Utility-privacy tradeoff (more noise = less accurate)
  - Complex parameter selection and management
  - Composition complexity for multiple queries
  - Not suitable for all types of data and analyses
  - Implementation complexity and potential bugs

##### DET006: Adversarial Training
- **Description**: Training models with adversarial examples to improve robustness
- **Applications**:
  - Image classification and object detection
  - Natural language processing and text classification
  - Speech recognition and audio processing
  - Anomaly detection and fraud prevention
  - Recommendation systems and personalization
- **Techniques**:
  - FGSM-based: Fast Gradient Sign Method
  - PGD-based: Projected Gradient Descent
  - Free adversarial training: Gradient ascent on loss
  - Trade-off constrained: Balancing natural and robust accuracy
  - Randomized smoothing: Certification through noise addition
- **Limitations**:
  - Increased training time and computational cost
  - Potential reduction in natural accuracy
  - Limited protection against unknown attack types
  - Requires generation or procurement of adversarial examples
  - May not transfer to different architectures or datasets
  - Possible negative impact on model interpretability

#### MITRE ATLAS vs OWASP LLM Top 10 Mapping
| MITRE ATLAS Tactic | OWASP LLM Top 10 Equivalent | Description |
|--------------------|-----------------------------|-------------|
| T016: Prompt Injection | LLM01:2023 - Prompt Injection | Direct and indirect prompt manipulation |
| T002: Data Source Identification / T012: Training Data Poisoning | LLM03:2023 - Training Data Poisoning | Data injection and corruption |
| T017: Evasion Attack | *(Not directly mapped)* | Test-time perturbation to evade detection |
| T018: Model Inversion | *(Related to LLM06)* | Reconstructing sensitive data from model |
| T019: Model Extraction | *(Related to LLM10)* | Stealing model functionality via API |
| T020: Adversarial ML via Third-Party Services | LLM07:2023 - Insecure Plugin Design | Compromised external ML services |
| T022: Inhibition | LLM04:2023 - Model Denial of Service | Resource exhaustion and DoS |
| T021: Exfiltration | LLM06:2023 - Sensitive Information Disclosure | Unauthorized data transfer |
| T023: Copyright/IP Circumvention | *(Related to LLM10)* | Unauthorized use of protected works |
| T024: Wrongful Deployment | *(Related to LLM09)* | Inappropriate model deployment |
| T006: Adversarial Crafting / T007: Backdoor Engineering | *(Related to LLM03)* | Backdoor insertion via data poisoning |

### NIST AI Risk Management Framework (AI RMF)
A voluntary framework for managing risks in AI systems

#### AI RMF 1.0 Structure
- **AI RMF Core**: Organizes outcomes into functions, categories, and subcategories
- **AI RMF Profiles**: Implementations of the Core for specific use cases
- **AI RMF Tiers**: Describe rigor of AI risk management practices
- **AI RMF Resources**: Supplementary materials and guidelines

#### Core Functions Deep Dive

##### GOVERN Function: Establish and implement the organization's AI risk management culture
**Categories**:
- **GOVERN-1: Policies, Procedures, and Processes**
  - Establish AI risk management policies
  - Define procedures for AI system lifecycle
  - Assign AI risk management roles and responsibilities
  - Develop AI risk management plans
  - Establish AI risk management training and awareness
- **GOVERN-2: Risk Management Strategy**
  - Define AI risk management approach and philosophy
  - Establish risk tolerance and thresholds
  - Define risk treatment preferences and strategies
  - Establish risk communication approaches
  - Define risk monitoring and reporting methodology
- **GOVERN-3: Awareness and Training**
  - Provide AI risk management literacy training
  - Conduct role-based training and exercises
  - Establish AI risk management awareness campaigns
  - Measure effectiveness of training programs
  - Update training based on feedback and changes
- **GOVERN-4: Third-Party Risk Management**
  - Establish third-party AI risk management policy
  - Define third-party AI risk assessment process
  - Monitor and manage third-party AI relationships
  - Conduct third-party AI due diligence and reviews
  - Establish termination and exit procedures for third-parties
- **GOVERN-5: Diversity, Inclusion, and Impartiality**
  - Address fairness, bias, and discrimination concerns
  - Promote diversity and inclusion in AI workforce
  - Implement impartiality practices in AI development
  - Address accessibility concerns for AI systems
  - Establish equity considerations for AI impacts
- **GOVERN-6: Legal and Regulatory Compliance**
  - Identify applicable laws and regulations for AI
  - Assess AI system compliance with laws and regulations
  - Manage legal and regulatory changes affecting AI
  - Document compliance efforts and evidence
  - Establish remediation processes for non-compliance

##### MAP Function: Establish context to inform AI-related risk management decisions
**Categories**:
- **MAP-1: Context Establishment**
  - Establish context for AI system development and use
  - Define AI system purpose and objectives
  - Identify stakeholders and their needs
  - Establish scope and boundaries for AI system
  - Identify applicable laws, regulations, and expectations
- **MAP-2: AI System Description**
  - Describe AI system architecture and components
  - Identify data sources and their characteristics
  - Describe model and algorithm selection rationale
  - Describe AI system limitations and assumptions
  - Identify dependencies and external factors
- **MAP-3: Risk Identification**
  - Identify benefits and costs of AI systems
  - Identify AI system vulnerabilities and limitations
  - Identify threats and hazard sources
  - Identify mitigation and existing controls
  - Identify impacts and consequences of AI systems
- **MAP-4: Impact Analysis**
  - Analyze legal and regulatory impacts of AI systems
  - Analyze financial and budgetary impacts of AI systems
  - Analyze societal and cultural impacts of AI systems
  - Analyze environmental impacts of AI systems
  - Analyze health and safety impacts of AI systems
- **MAP-5: Tradeoff Analysis**
  - Analyze tradeoffs between AI system characteristics
  - Analyze tradeoffs between benefits and costs
  - Analyze tradeoffs between risks and mitigations
  - Analyze tradeoffs between use and non-use of AI
  - Analyze tradeoffs between alternative approaches

##### MEASURE Function: Employ appropriate techniques to identify, analyze, and measure AI-related risks
**Categories**:
- **MEASURE-1: Analysis and Test Bench**
  - Identify appropriate analysis and testing methods
  - Develop AI system test plans and methodologies
  - Acquire and prepare necessary resources for testing
  - Conduct AI system testing and evaluation
  - Analyze test results and identify issues
- **MEASURE-2: Monitoring**
  - Establish AI system monitoring approach
  - Implement AI system monitoring and logging
  - Detect anomalies and unexpected behavior
  - Track performance and trends over time
  - Validate AI system behavior and outputs
- **MEASURE-3: Data and Privacy**
  - Identify data sources and their characteristics
  - Assess data quality and suitability for AI
  - Evaluate privacy impacts of AI systems
  - Assess bias and fairness impacts of AI systems
  - Evaluate data provenance and lineage
- **MEASURE-4: Resources**
  - Identify resources necessary for AI system function
  - Assess resource adequacy and availability
  - Evaluate resource efficiency and utilization
  - Track resource dependencies and changes
  - Validate resource specifications and quality
- **MEASURE-5: Impact Assessment**
  - Apply impact assessment methods to AI systems
  - Document assumptions and limitations of assessments
  - Track impacts over time and changes
  - Evaluate assessment quality and effectiveness
  - Integrate assessment findings into decision-making

##### MANAGE Function: Allocate resources to address AI-related risks
**Categories**:
- **MANAGE-1: Planning**
  - Plan resources for AI system risk management
  - Establish risk treatment priorities and plans
  - Develop risk treatment strategies and approaches
  - Assign resources for risk treatment activities
  - Validate risk treatment plans and approaches
- **MANAGE-2: Preemptive**
  - Apply risk treatment strategies to prevent risks
  - Manage resources for risk treatment activities
  - Track risk treatment implementation and progress
  - Validate risk treatment effectiveness and outcomes
- **MANAGE-3: Recovered**
  - Apply risk treatment strategies to respond to risks
  - Manage resources for risk treatment activities
  - Track risk treatment implementation and progress
  - Validate risk treatment effectiveness and outcomes
- **MANAGE-4: Respond and Recover**
  - Respond to AI system incidents and events
  - Manage resources for incident response activities
  - Track incident response implementation and progress
  - Validate incident response effectiveness and outcomes
  - Recover from AI system incidents and events

#### AI RMF Tiers (Describe Rigor of Risk Management)
- **Tier 1: Partial** 
  - Risk management processes are not formalized
  - Risk management is reactive and inconsistent
  - Limited awareness of AI risks at organizational level
  - Limited information sharing about AI risks
- **Tier 2: Risk Informed**
  - Risk management processes approved by management
  - Risk management is regularly but not consistently applied
  - Organizational risk management policy exists
  - Information sharing about AI risks occurs
- **Tier 3: Repeatable**
  - Formalized risk management policies exist
  - Risk management is consistently applied
  - Organization-wide risk management approach exists
  - Resources for risk management are sufficient
- **Tier 4: Adaptive**
  - Adaptive and responsive risk management approach
  - Continuous improvement through learning from errors
  - Risk management is part of organizational culture
  - Resources for risk management are committed based on risk

#### AI RMF Profiles (Use-Case Specific Implementations)
- **HR and Employment Technology Profile**:
  - Resume screening and candidate matching
  - Employee performance evaluation and feedback
  - Workplace monitoring and productivity tracking
  - Promotion and succession planning
  - Workforce planning and skills analysis
- **Finance and Banking Profile**:
  - Credit scoring and lending decisions
  - Fraud detection and prevention
  - Algorithmic trading and investment advice
  - Risk assessment and portfolio management
  - Regulatory reporting and compliance
- **Healthcare and Medical Profile**:
  - Medical diagnosis and treatment recommendations
  - Medical imaging analysis and interpretation
  - Drug discovery and development
  - Clinical trial design and patient selection
  - Hospital operations and resource management
- **Law Enforcement and Justice Profile**:
  - Predictive policing and resource allocation
  - Risk assessment and decision support
  - Forensic analysis and evidence processing
  - Court administration and case management
  - Rehabilitation and recidivism reduction
- **Education and Learning Profile**:
  - Adaptive learning and personalized instruction
  - Plagiarism detection and academic integrity
  - Automated grading and feedback
  - Educational assessment and testing
  - Educational resource recommendation
- **Autonomous Vehicles Profile**:
  - Perception and object detection
  - Path planning and navigation
  - Control and actuation systems
  - Safety systems and collision avoidance
  - User experience and interaction
- **Manufacturing and Industrial Profile**:
  - Quality control and defect detection
  - Predictive maintenance and equipment failure
  - Supply chain optimization and logistics
  - Robotics and automation systems
  - Energy management and resource efficiency
- **Retail and E-commerce Profile**:
  - Product recommendation and personalization
  - Demand forecasting and inventory management
  - Chatbots and customer service automation
  - Price optimization and dynamic pricing
  - Supply chain and logistics optimization
- **Energy and Utilities Profile**:
  - Smart grid management and optimization
  - Renewable energy forecasting and integration
  - Energy efficiency and conservation measures
  - Asset management and maintenance optimization
  - Grid security and threat detection
- **Telecommunications Profile**:
  - Network traffic analysis and optimization
  - Churn prediction and customer retention
  - Fraud detection and prevention
  - Service quality and performance monitoring
  - Resource allocation and optimization

#### Implementation Guidance for AI RMF
**Getting Started**:
1. **Leadership Commitment**: Obtain executive sponsorship and resources
2. **Gap Analysis**: Compare current state to AI RMF desired outcomes
3. **Priority Setting**: Determine which functions/categories to address first
4. **Resource Allocation**: Assign personnel, budget, and time
5. **Implementation Planning**: Develop roadmap and milestones
6. **Training and Awareness**: Educate workforce on AI RMF concepts
7. **Pilot Implementation**: Start with a specific AI system or use case
8. **Scaling and Expansion**: Apply lessons learned to other systems
9. **Continuous Improvement**: Regularly review and improve implementation

**Integration with Existing Frameworks**:
- **NIST Cybersecurity Framework (CSF)**: 
  - MAP → Identify
  - MEASURE → Protect + Detect
  - GOVERN + MANAGE → Respond + Recover
  - Shared concepts: risk management, continuous improvement
- **ISO 27001**: 
  - GOVERN → Clause 5 (Leadership) + Clause 6 (Planning)
  - MAP → Clause 8.2 (Risk Assessment)
  - MEASURE → Clause 8.1 (Operational Planning) + Clause 9 (Performance Evaluation)
  - MANAGE → Clause 8.3 (Risk Treatment) + Clause 10 (Improvement)
- **SOC 2**: 
  - Security criteria align with GOVERN, MEASURE, MANAGE
  - Availability criteria align with MEASURE, MANAGE
  - Processing integrity criteria align with MEASURE
  - Confidentiality criteria align with MEASURE
- **ISO 22301 (Business Continuity)**:
  - MANAGE-4 (Respond and Recover) aligns with business continuity
  - GOVERN-1 (Policies) aligns with BCMS establishment
  - MAP-2 (Impact Analysis) aligns with BIA and RA
  - MEASURE-2 (Monitoring) aligns with BCE and monitoring

#### AI RMF and Other AI-Specific Frameworks
- **AI RMF vs OECD AI Principles**:
  - Both emphasize human-centered values and fairness
  - AI RMF more operational and implementation-focused
  - OECD more principle-based and policy-oriented
- **AI RMF vs EU AI Act**:
  - AI RMF voluntary framework for risk management
  - EU AI Act regulatory requirement for high-risk AI
  - AI RMF can help achieve compliance with AI Act
  - Different scopes and enforcement mechanisms
- **AI RMF vs NIST CSF for Critical Infrastructure**:
  - AI RMF broader scope (all AI systems)
  - NIST CSF more focused on critical infrastructure
  - Complementary rather than competing
  - Can be used together for comprehensive coverage

#### Practical Implementation Examples
**Simple AI RMF Implementation Checklist**:
```
[ ] GOVERN-1: Established AI risk management policy
[ ] GOVERN-2: Defined risk management strategy and tolerance
[ ] GOVERN-3: Implemented workforce training program
[ ] GOVERN-4: Established third-party risk management process
[ ] GOVERN-5: Addressed diversity, inclusion, and impartiality
[ ] GOVERN-6: Documented legal and regulatory compliance
[ ] MAP-1: Established context and purpose for AI systems
[ ] MAP-2: Documented AI system architecture and data sources
[ ] MAP-3: Identified AI-specific risks and threats
[ ] MAP-4: Analyzed legal, financial, societal, environmental impacts
[ ] MAP-5: Analyzed tradeoffs and alternative approaches
[ ] MEASURE-1: Established testing and evaluation methodologies
[ ] MEASURE-2: Implemented monitoring and logging systems
[ ] MEASURE-3: Assessed data quality, privacy, bias, and fairness
[ ] MEASURE-4: Tracked resource utilization and dependencies
[ ] MEASURE-5: Documented impact assessments and limitations
[ ] MANAGE-1: Planned risk treatment strategies and resources
[ ] MANAGE-2: Implemented preventive risk treatments
[ ] MANAGE-3: Implemented responsive risk treatments
[ ] MANAGE-4: Established incident response and recovery plans
```

**Sample AI System GOVERN Artifacts**:
- **AI Risk Management Policy**: 2-5 page document stating commitment
- **Roles and Responsibilities Matrix**: RACI chart for AI activities
- **Risk Register**: Spreadsheet or database tracking AI risks
- **Training Curriculum**: Slides, videos, exercises for workforce
- **Third-Party Assessment Questionnaire**: Security and privacy questions
- **Compliance Tracking Spreadsheet**: Laws, regulations, status

**Sample AI System MAP Artifacts**:
- **System Architecture Diagram**: Components, data flows, interfaces
- **Data Dictionary**: Descriptions of data sources, fields, types
- **Model Card**: Metadata about model purpose, performance, limitations
- **Data Sheet**: Metadata about training data collection, processing
- **Assumptions and Limitations Document**: Known constraints and boundaries
- **Stakeholder Analysis Matrix**: Needs, influence, engagement strategies

**Sample AI System MEASURE Artifacts**:
- **Test Plan**: Objectives, scope, resources, procedures, acceptance criteria
- **Monitoring Dashboard**: Real-time views of key metrics and health
- **Data Quality Report**: Completeness, accuracy, consistency, validity
- **Bias and Fairness Report**: Disparate impact analysis, subgroup performance
- **Resource Utilization Report**: CPU, memory, disk, network over time
- **Impact Assessment Document**: Legal, financial, societal, environmental

**Sample AI System MANAGE Artifacts**:
- **Risk Treatment Plan**: Prioritized actions, resources, timeline, owners
- **Incident Response Playbook**: Detection, containment, eradication, recovery
- **Backup and Recovery Procedure**: RPO, RTO, procedures, verification
- **Business Continuity Plan**: Critical functions, alternate sites, communication
- **Lessons Learned Repository**: Documented improvements and changes
```

## Framework Comparison and Selection Guide

### When to Use Each Framework

#### Use PLOT4AI When:
- Focusing specifically on privacy, legal, operational, and technical risks
- Need for comprehensive risk categorization in AI systems
- Early-stage risk identification and categorization
- Complementing other frameworks with AI-specific perspectives
- Teaching and training on AI risk domains

#### Use OWASP LLM Top 10 When:
- Building or securing LLM-specific applications
- Need for concrete, actionable security controls
- Development and deployment lifecycle security
- Penetration testing and vulnerability assessment
- Developer and DevSecOps team guidance
- Compliance with application security standards

#### Use MITRE ATLAS When:
- Understanding adversarial ML tactics and techniques
- Threat intelligence sharing and collaboration
- Attack surface analysis and reduction
- Defensive strategy development and prioritization
- Red team/blue team exercises and simulation
- Incident response and forensic analysis
- Research and development of ML security

#### Use NIST AI RMF When:
- Comprehensive AI risk management program development
- Organizational-level risk management implementation
- Regulatory compliance and demonstration
- Integration with existing risk management frameworks
- Executive reporting and governance
- Continuous improvement and maturity modeling
- Federal contractor requirements (increasingly common)

### Framework Complementarity
**Combined Approach for Comprehensive AI Security**:
1. **Start with NIST AI RMF** for overall governance structure
2. **Use PLOT4AI** for detailed risk categorization within MAP and MEASURE
3. **Apply OWASP LLM Top 10** for LLM-specific technical controls in MEASURE and MANAGE
4. **Leverage MITRE ATLAS** for threat modeling and defensive planning in MAP and MANAGE
5. **Integrate with existing frameworks** (CSF, ISO 27001, etc.) for organizational alignment

### Practical Implementation Roadmap
**Phase 1: Foundation (Months 1-3)**
- Executive sponsorship and resource allocation
- NIST AI RMF gap analysis and planning
- Basic policies and procedures establishment
- Initial workforce training and awareness
- Inventory of AI systems and use cases

**Phase 2: Risk Assessment (Months 4-6)**
- Detailed MAP function implementation
- PLOT4AI-based risk categorization
- OWASP LLM Top 10 technical assessment
- MITRE ATLAS threat modeling
- Risk register creation and prioritization

**Phase 3: Control Implementation (Months 7-9)**
- MEASURE function implementation (testing, monitoring)
- MANAGE function implementation (risk treatment)
- OWASP LLM Top 10 control deployment
- MITRE ATLAS defensive countermeasures
- Continuous monitoring and alerting setup

**Phase 4: Monitoring and Improvement (Months 10-12)**
- Ongoing MEASURE function activities
- Incident response and MANAGE-4 activation
- Framework effectiveness measurement
- Continuous improvement and updates
- Preparation for next cycle

### Recommended Study Resources
**PLOT4AI**:
- Official website: https://plot4ai.org/
- White papers and case studies
- Community forums and discussions
- Academic publications on AI risk domains

**OWASP LLM Top 10**:
- Official OWASP page: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Cheat sheets and quick reference guides
- Prevention and mitigation guides
- Tools and utilities for testing
- Community projects and extensions

**MITRE ATLAS**:
- Official MITRE ATLAS: https://atlas.mitre.org/
- ATT&CK framework knowledge (https://attack.mitre.org/)
- D3FEND framework knowledge (https://d3fend.mitre.org/)
- CALDERA for adversary emulation
- Atomic Red Team for testing techniques
- CAR Analytics for data models

**NIST AI RMF**:
- Official NIST page: https://www.nist.gov/itl/ai-risk-management-framework
- AI RMF 1.0 document (NIST AI 100-1)
- AI RMF Playbook (NIST AI 100-5)
- AI RMF Explainer Series
- Companion resources and implementation guides
- Industry-specific profiles and overlays

**Integration Resources**:
- NIST CSF: https://www.nist.gov/cyberframework
- ISO 27001: https://www.iso.org/standard/54534.html
- SOC 2: https://www.aicpa.org/
- ISO 22301: https://www.iso.org/standard/75106.html
- COBIT: https://www.isaca.org/resources/cobit