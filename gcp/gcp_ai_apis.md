# Google Cloud Platform AI/ML APIs & Services

## Core AI/ML Platform Services

### Vertex AI Platform
- **aiplatform.googleapis.com** - Google Cloud Vertex AI Platform
- **automl.googleapis.com** - AutoML API for custom model training

### Conversational AI
- **dialogflow.googleapis.com** - Dialogflow for chatbots and voice assistants
- **contactcenteraiplatform.googleapis.com** - Contact Center AI Platform

### Document AI
- **documentai.googleapis.com** - Document AI for document processing and analysis

### Cloud AI Companion
- **cloudaicompanion.googleapis.com** - Cloud AI Companion services

## Pre-trained AI APIs

### Vision & Image Analysis
- **vision.googleapis.com** - Cloud Vision API for image analysis
- **videointelligence.googleapis.com** - Video Intelligence API

### Natural Language Processing
- **language.googleapis.com** - Natural Language API
- **translate.googleapis.com** - Translation API

### Speech & Audio
- **speech.googleapis.com** - Speech-to-Text API
- **texttospeech.googleapis.com** - Text-to-Speech API

## Third-Party AI Services

### Anthropic Claude
- **claude-3-5-haiku.endpoints.mp-anthropic.cloud.goog**
- **claude-3-haiku.endpoints.mp-anthropic.cloud.goog**
- **claude-3-provisioned-throughput.cloudpartnerservices.goog**

### AI21 Studio
- **ai21-studio-saas.endpoints.ai21-public.cloud.goog**
- **ai21-studio.endpoints.ai21-public.cloud.goog**

### Deepgram
- **deepgram-language-ai.endpoints.deepgram-public.cloud.goog**

### ElevenLabs
- **elevenlabs-tts-and-audio-ai-platform.endpoints.elevenlabs-public.cloud.goog**

## MLOps & Model Management

### DataRobot
- **datarobot-ai-cloud-platform-for-google-cloud.endpoints.datarobot-public.cloud.goog**

### Comet ML
- **comet-mlops-platform.endpoints.comet-vm-public.cloud.goog**

### Arize AI
- **arize-ai.endpoints.arize-public.cloud.goog**

### DataLoop
- **dataloop-ai.endpoints.dataloop-public.cloud.goog**

## Business Intelligence & Analytics

### C3 AI Suite
- **c3-ai-cash-management-c3ai-marketplace.cloudpartnerservices.goog**
- **c3-ai-crm-c3ai-marketplace.cloudpartnerservices.goog**
- **c3-ai-energy-management-c3ai-marketplace.cloudpartnerservices.goog**
- **c3-generative-ai-for-documents.endpoints.c3ai-marketplace.cloud.goog**

### Quantiphi BaionIQ
- **baioniq.endpoints.quantiphi-public-376012.cloud.goog**

### DataChat
- **datachat.endpoints.datachat-ai.cloud.goog**

## Industry-Specific AI Solutions

### Healthcare & Medical
- **aiforia-clinical-suites.endpoints.aiforia-public.cloud.goog**
- **ai-orchestrator.endpoints.ferrumhealth-public.cloud.goog**

### Retail & E-commerce
- **antavo-ai-loyalty-cloud.endpoints.antavo-public.cloud.goog**
- **contract-compliance.endpoints.traxretail-public.cloud.goog**

### Financial Services
- **deloitte-aml-investigation-agent-standalone.endpoints.us-con-gcp-sbx-0000427-020625.cloud.goog**
- **aishield-ai-security-platform.endpoints.rbprj-100113.cloud.goog**

### Manufacturing & Industrial
- **avathon-industrial-ai-platform.endpoints.avathon-public.cloud.goog**
- **brainos-sense-suite.endpoints.braincorp-public1.cloud.goog**

## Security & Compliance

### AI Security
- **ai-runtime-security.endpoints.paloaltonetworks-public.cloud.goog**
- **aishield-guardian.endpoints.rbprj-100113.cloud.goog**
- **aporia-guardrails.endpoints.aporia-public.cloud.goog**

### Document Security
- **document-forensics-saas.endpoints.resistant-ai-public.cloud.goog**
- **document-forensics.endpoints.resistant-ai-public.cloud.goog**

## Developer Tools & Platforms

### Code Generation & Development
- **augment-code-developer-ai-for-teams.endpoints.augmentcomputing-public.cloud.goog**
- **ask-ai.endpoints.ask-ai-public.cloud.goog**

### Low-Code/No-Code AI
- **autoql-by-chata.ai.endpoints.chataai-public.cloud.goog**
- **aible.endpoints.aible-gcp-marketplace-public.cloud.goog**

## AI Infrastructure & Deployment

### Model Serving
- **bentocloud.endpoints.bentoml-public.cloud.goog**
- **cserve.endpoints.centml-public.cloud.goog**

### Confidential AI
- **cosmian-confidential-ai-rhel-9-amd-sev-snp.endpoints.cosmian-public.cloud.goog**
- **cosmian-confidential-ai-ubuntu-22.amd-sev-snp.endpoints.cosmian-public.cloud.goog**

## Communication & Customer Service

### Contact Centers
- **contact-center-quality-platform.endpoints.yoshai-public.cloud.goog**
- **callzen.ai.endpoints.nobroker-callzen-public.cloud.goog**

### Voice & Audio
- **ai-voice-generator.endpoints.resembleai-public.cloud.goog**
- **resembleai-voice-cloning.endpoints.resembleai-public.cloud.goog**

## Marketing & Advertising

### Marketing Intelligence
- **deloitte-marketing-content-and-campaign-agent-suite.endpoints.us-con-gcp-sbx-0000427-020625.cloud.goog**
- **anonymized-social-ad-rates-for-sports-entertainment.endpoints.blinkfire-ai.cloud.goog**

### Ad Connectors (Windsor.ai)
- **facebook-ads-connector-by-windsor.ai.endpoints.bigquery-connectors-public.cloud.goog**
- **google-ads-connector-by-windsor.ai.endpoints.bigquery-connectors-public.cloud.goog**
- **amazon-ads-connector-by-windsor.ai.endpoints.bigquery-connectors-public.cloud.goog**

## Getting Started

### Essential APIs to Enable First:
1. **aiplatform.googleapis.com** - Core Vertex AI platform
2. **dialogflow.googleapis.com** - Conversational AI
3. **vision.googleapis.com** - Image analysis
4. **language.googleapis.com** - Text analysis
5. **speech.googleapis.com** - Speech-to-text
6. **translate.googleapis.com** - Translation services

### Enable APIs Command:
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable dialogflow.googleapis.com
gcloud services enable vision.googleapis.com
gcloud services enable language.googleapis.com
gcloud services enable speech.googleapis.com
gcloud services enable translate.googleapis.com
```

### Authentication Setup:
```bash
# Set up application default credentials
gcloud auth application-default login

# For service account authentication
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

---

*File saved at: `/home/gouru1sri/gcp_ai_apis.md`*