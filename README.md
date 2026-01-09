# rag-complaint-chatbot
Below is a **complete, submission-ready `README.md`** that accurately reflects **Tasks 1–4** and **exactly what you implemented**, based on our discussion and your notebook work.
You can copy this directly into your repository.

---

# CFPB Complaint RAG Chatbot

## Project Overview

This project builds a **Retrieval-Augmented Generation (RAG)** system using real consumer complaint data from the **Consumer Financial Protection Bureau (CFPB)**. The goal is to enable users to ask natural-language questions about customer complaints across multiple financial products and receive **grounded, explainable answers** supported by retrieved complaint excerpts.

The system combines:

* Exploratory data analysis and preprocessing
* Semantic embeddings and vector search
* Prompt-guided large language model (LLM) generation
* An interactive user interface for non-technical users

---

## Dataset

The project uses CFPB consumer complaint data, which includes:

* Structured metadata (product, company, issue, state, dates)
* Unstructured free-text **Consumer complaint narratives**

Two datasets are used:

1. **Full CFPB complaint dataset** (`complaints.csv`) for EDA and preprocessing
2. **Pre-built embeddings dataset** (`complaint_embeddings.parquet`) for full-scale retrieval in the RAG pipeline

---

## Project Structure

```
rag-complaint-chatbot/
├── data/
│   ├── raw/
│   └── processed/
├── vector_store/              # Persisted ChromaDB index
├── notebooks/                 # EDA, preprocessing, and experimentation
├── src/
│   └── rag_pipeline.py        # Reusable RAG core logic
├── app.py (optional)          # UI script if run outside notebooks
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Task 1: Exploratory Data Analysis and Preprocessing

### Objective

Understand the structure, quality, and content of the CFPB complaint data and prepare it for downstream embedding and retrieval.

### Key Steps

* Loaded the full CFPB complaint dataset (`complaints.csv`)
* Examined dataset shape, column types, and missing values
* Analyzed the distribution of complaints across products
* Computed and visualized word counts of complaint narratives
* Identified complaints with and without narratives

### Filtering

To meet project requirements:

* Retained complaints for the following product categories:

  * Credit Card
  * Personal Loan
  * Savings Account
  * Money Transfers
* Removed records with missing or empty complaint narratives

### Text Cleaning

To improve embedding quality:

* Converted text to lowercase
* Removed boilerplate phrases (e.g., “I am writing to file a complaint…”)
* Removed special characters and excessive whitespace

The cleaned dataset was saved for downstream tasks.

---

## Task 2: Chunking, Embedding, and Vector Store Indexing

### Objective

Convert complaint narratives into a format suitable for efficient semantic search.

### Stratified Sampling

* Created a stratified sample of **10,000–15,000 complaints**
* Ensured proportional representation across all selected product categories
* Used this subset to prototype chunking and embedding pipelines efficiently

### Text Chunking

* Implemented a chunking strategy to avoid embedding long narratives as single vectors
* Used fixed-size chunks with overlap to preserve context
* Final configuration:

  * Chunk size: ~500 characters
  * Chunk overlap: ~50 characters

### Embedding Model

* Selected: `sentence-transformers/all-MiniLM-L6-v2`
* Rationale:

  * Strong semantic performance
  * Small embedding size (384 dimensions)
  * Efficient inference on CPU environments

### Vector Store

* Used **ChromaDB** to store embeddings
* Stored metadata alongside each vector, including:

  * Complaint ID
  * Product category
  * Company
  * Issue and sub-issue
  * Date received
* Persistence handled automatically via ChromaDB’s persistent directory

---

## Task 3: Building the RAG Core Logic and Evaluation

### Objective

Build a complete retrieval-and-generation pipeline using the full-scale pre-built vector store.

### Retrieval

* Loaded the pre-built ChromaDB vector store containing embeddings for the full dataset
* Implemented a retriever that:

  * Embeds the user query using the same embedding model
  * Performs similarity search
  * Retrieves the top-k most relevant complaint chunks (k = 5)

### Prompt Engineering

A strict prompt template was designed to enforce grounded answers:

```
You are a financial analyst assistant for CrediTrust.
Answer the question using ONLY the complaint excerpts provided below.
If the information is not present in the context, say you do not have enough information.
```

This minimizes hallucinations and opinionated responses.

### Generation

* Combined retrieved chunks and user query into a single prompt
* Generated answers using an instruction-tuned LLM
* Implemented context truncation to respect model input limits

### Modularization

* Refactored all RAG logic into a reusable Python module: `src/rag_pipeline.py`
* Exposed a clean interface for running queries through the pipeline
* This improved maintainability, reproducibility, and reuse across evaluation and UI layers

### Qualitative Evaluation

* Created a set of representative user questions
* Ran each question through the RAG pipeline
* Evaluated results using:

  * Answer quality
  * Relevance of retrieved sources
  * Groundedness
* Documented findings in an evaluation table in the report

---

## Task 4: Interactive Chat Interface

### Objective

Enable non-technical users to interact with the RAG system.

### Implementation

* Built an interactive interface using **Gradio**, executed directly within a Jupyter notebook
* Core features:

  * Text input for user questions
  * “Ask” button to submit queries
  * Display area for AI-generated answers
  * Display of retrieved source excerpts below each answer
  * “Clear” button to reset the interface

### Trust and Usability

* Displaying retrieved sources increases transparency and user trust
* Users can verify that answers are grounded in real CFPB complaints
* The interface supports rapid experimentation and demonstration

---

## Key Learning Outcomes

Through this project, we:

* Built an end-to-end RAG system on real-world, unstructured data
* Learned how to design and query a vector database for semantic search
* Applied prompt engineering to reduce hallucinations
* Balanced performance constraints with practical engineering choices
* Designed a user-friendly interface for applied NLP systems

---

## How to Run

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Open the notebooks to explore:

   * EDA and preprocessing
   * Chunking and embedding
   * RAG evaluation

3. Launch the Gradio interface directly from the notebook to interact with the chatbot.

---

## Notes

* ChromaDB persistence is handled automatically
* The system is designed to be modular and extensible
* The project structure supports both experimentation and production-style reuse

