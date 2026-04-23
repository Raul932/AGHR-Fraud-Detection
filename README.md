# AGHR Fraud Detection: Multi-Agent AI System

This repository contains an advanced, modular AI pipeline developed for the **Reply AI Agents Challenge 2026**. The system is designed to detect sophisticated "Mirror Hackers" within a simulated banking ecosystem (MirrorPay) by orchestrating specialized AI agents through a state-machine architecture.

## 🚀 Key Features

* **Multimodal Intelligence:** Automated transcription of vishing (voice phishing) calls using **Google Gemini 1.5 Pro** to identify extortion and social engineering triggers.
* **Agentic Orchestration:** A **LangGraph-powered** state machine that manages four specialized expert personas (Financial, Identity, NLP, and Meta-Reviewer).
* **Semantic RAG (Retrieval-Augmented Generation):** Integration with **ChromaDB** to retrieve similar historical fraud patterns, providing agents with few-shot context.
* **High-Signal Feature Engineering:** Custom Python logic for calculating **Haversine distances** (geo-anomalies), **Z-scores** (financial outliers), and transaction velocity.
* **Production Observability:** Full integration with **Langfuse** for trace tracking, cost analysis, and latency monitoring.
* **Optimization Layer:** SHA-256 based **Risk Score Caching** to minimize redundant LLM calls and preserve token budget.

---

## 🏗️ System Architecture

The pipeline is divided into 5 distinct phases to ensure separation of concerns and scalability:

### Phase 1: Data Enrichment & Audio Transcription
* **`phase1_preprocessing.py`**: Ingests raw transaction data and performs complex calculations (Velocity 24h, Z-score, Geo-distance) to label potential risks before agents evaluate them.
* **`phase1b_audio.py`**: Handles binary audio files, converting vishing calls into structured text logs for sentiment and intent analysis.

### Phase 2: Memory & Context (RAG)
* **`phase2_rag_cache.py`**: Vectorizes enriched records into a persistent **ChromaDB** collection. It uses semantic search to find "look-alike" fraudulent cases from the training set.

### Phase 3: The Multi-Agent "Brain"
* **`phase3_langgraph.py`**: Defines the parallel execution graph:
    1.  **Financial Agent**: Analyzes probing payments, balance ratios, and statistical outliers.
    2.  **Identity Agent**: Detects geographical anomalies and "Night Prowling" behavioral shifts.
    3.  **NLP Agent**: Correlates transactions with phishing history and vishing transcripts.
    4.  **Meta-Reviewer**: Acts as the "Supreme Judge," resolving conflicts between specialists to maximize **Recall**.

### Phase 4 & 5: Expert Prompting & Export
* **`phase4_prompts.py`**: Highly-tuned system instructions designed to force structured JSON outputs and maintain strict expert personas.
* **`phase5_export.py`**: Aggregates multi-agent results, flushes traces to Langfuse, and generates the final submission files.

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **AI Framework** | LangChain / LangGraph |
| **Vector DB** | ChromaDB (Local Persistent) |
| **Models** | GPT-4o-mini, DeepSeek-V3, Gemini 1.5 Pro |
| **Analysis** | Pandas, NumPy |
| **Observability** | Langfuse |

---

## 🏁 Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install langchain langgraph chromadb pandas requests python-dotenv ulid-py langfuse
    ```

2.  **Configuration:**
    Create a `.env` file and add your credentials:
    ```text
    OPENROUTER_API_KEY=your_key
    LANGFUSE_PUBLIC_KEY=your_key
    LANGFUSE_SECRET_KEY=your_key
    TEAM_NAME=YourTeamName
    ```

3.  **Run the Pipeline:**
    ```bash
    python main.py --data-dir "The Truman Show - train" --world-name "truman"
    ```

4.  **Diagnostic Tool:**
    ```bash
    python analyze_data.py
    ```

---

## 📈 Performance Tracking
This project uses **Langfuse** to monitor agent performance. Each run generates a unique Session ID, allowing for granular review of how the Meta-Reviewer arrived at a specific fraud verdict based on agent consensus.

---
**Disclaimer:** This system was built for the Reply AI Agents Challenge and is optimized for the MirrorPay simulation dataset.
