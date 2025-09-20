# Research Automation with Multi-Agent RAG and LoRA

## Project Overview  
This project builds an **end-to-end research automation pipeline** that transforms raw scientific literature into structured, high-quality academic reports. It integrates **retrieval-augmented generation (RAG)**, **vector database search**, **multi-agent collaboration**, and **LoRA fine-tuning**, ensuring both scientific rigor and domain-specific adaptability.  

Key contributions:  
- Automates the pipeline from **literature ingestion → retrieval → report drafting → multi-agent review → fine-tuning**  
- Provides **domain fidelity** for electrochemistry and CO₂ reduction research (custom formula processor & reference standardization)  
- Supports **HPC deployment** with GPU optimization and robust logging  

---

## Core Features  

### Literature Processing & Embedding  
- Parse academic papers into structured Markdown  
- Extract metadata (title, authors, year, abstract)  
- Preserve LaTeX/math formulas with a custom formula processor  
- Store document chunks in a **FAISS vector database** using HuggingFace embeddings  

### RAG-based Draft Generation  
- Retrieve relevant literature context from FAISS  
- Generate structured reports with sections:  
  - **Background Knowledge**  
  - **Current Research**  
  - **Research Recommendations**  
- Integrate **CrossRef API** for standardized citations  

### Multi-Agent Collaboration  
- **Technical Expert**: verifies scientific mechanisms, formulas, and units  
- **Literature Expert**: improves readability, coherence, and logical flow  
- **Data Validator**: checks numerical plausibility and citation accuracy  
- **Captain Agent**: consolidates all reviews into a final version  
- Iterative refinement loop until all quality scores pass threshold  

### LoRA Fine-Tuning  
- Collect improved outputs from the multi-agent system  
- Apply **Low-Rank Adaptation (LoRA)** for efficient fine-tuning of large language models  
- Boosts domain performance while keeping computational costs low  

### Deployment & HPC Compatibility  
- Runs on **local or cluster GPUs** (A100, H100, etc.)  
- Configurable resource allocation, logging (`.out` / `.err`), and debug mode  
- Integrated with **Ollama** for local LLM inference  

---

## Tech Stack  

- **Languages**: Python 3.10+  
- **Core Libraries**:  
  - [LangChain](https://github.com/langchain-ai/langchain)  
  - [FAISS](https://github.com/facebookresearch/faiss)  
  - [SentenceTransformers](https://www.sbert.net/)  
  - [HuggingFace Transformers](https://huggingface.co/transformers/)  
  - [PEFT / LoRA](https://huggingface.co/docs/peft/index)  
  - [AG2 (AutoGen v2)](https://github.com/microsoft/autogen)  
- **Model Backend**: Qwen (via [Ollama](https://ollama.com/))  
- **External APIs**: CrossRef  

---

## Quick Start  

```bash
# Clone repository
git clone https://github.com/Zhangkanghao11/Multi-Agent

# Install dependencies
pip install -r environment.yml

# Step 1: Build embeddings from literature
python step1.py

# Step 2: Generate draft chapters with RAG
python step2.py

# Step 3: Multi-agent review & iterative refinement
python step3.py

# Step 4: Mermaid Figure
python step4.py

# Fine-tune model with LoRA (optional)
python fine_tune.py
