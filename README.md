# 🚀 GDPR-Aligned Automated Redaction System for Judicial Documents  
### *A Privacy-Preserving Retrieval-Augmented Generation (RAG) Pipeline for Courts*

This repository contains a GDPR-compliant automated redaction system designed for **judicial document processing**.  
The system uses a **Retrieval-Augmented Generation (RAG)** architecture with strict privacy controls to ensure that personally identifiable information (PII) in court transcripts is **never exposed**, even when queried directly.

This project includes:

- ✅ A **Streamlit web UI** for live interaction  
- ✅ A **backend redaction-aware RAG engine**  
- ✅ A **synthetic dataset of judicial transcripts with embedded PII**  
- ✅ A **test harness** validating summarization, redaction, and refusal behavior  

Built for educational and research use, this system demonstrates how AI can support courts **while remaining aligned with GDPR**, FOIA Exemption 6, and privacy-by-design principles.

---


# 🔧 System Overview

## 🔹 Architecture Layers

1. **PDF → Text Extraction** using PyMuPDF  
2. **Deterministic text chunking** (300 words, with overlap)  
3. **Embedding generation** using Azure OpenAI’s `text-embedding-3-small`  
4. **Local FAISS vector index** (non-persistent, GDPR-friendly)  
5. **Privacy Enforcement Layer**
   - Regex detection of PII-seeking queries  
   - Redaction of PII using `[REDACTED]`  
   - Refusal messages for prohibited requests  
6. **GPT-4o language model** with strict system prompts  
7. **Streamlit UI** for real-time interaction  

This architecture is explicitly aligned with **GDPR Article 25 – Privacy by Design & Default**.

---

# 🔒 PII Protection Modes

The system enforces three privacy behaviors:

### **1. Summarization Mode**
Provides high-level summaries containing **zero PII**.

### **2. Redaction Mode**
Replaces PII in retrieved text with **`[REDACTED]`** before sending it to the model.

### **3. Refusal Mode**
Direct PII extraction attempts always return:

> **“I’m sorry, but I can’t share that information.”**

Protection is enforced via:
- Regex-based harmful query detection  
- Redaction preprocessing  
- Retrieval minimization  
- Strict system prompt constraints  

---

# 🖥️ Streamlit UI (Interactive)

The Streamlit UI (`app.py`) allows you to:

- Upload PDF transcripts  
- Ask questions live  
- Determine how the system redacts or refuses  
- Optionally display retrieved chunks for debugging  

### 📦 Running the UI

```bash
# Before running the UI, install all required Python modules:

pip install streamlit pymupdf faiss-cpu python-dotenv openai numpy

# Run the UI:

streamlit run app.py
