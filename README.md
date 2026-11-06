# Pharmaceutical Document Q&A with Retrieval-Augmented Generation (RAG)

A document-grounded question-answering prototype built in **Google Colab** using **Python, LlamaIndex, and Google Gemini**. The system extracts text from pharmaceutical PDFs, retrieves relevant passages, reranks them, and uses those passages to answer questions about product quality and storage requirements.

Developed as part of the **Pfizer × Extern AI-Powered Document Intelligence externship**.

## Project overview

Large language models can produce plausible answers that are not supported by a particular document. This project explores **Retrieval-Augmented Generation (RAG)**: retrieving relevant text from the source PDF *before* asking an LLM to answer.

The prototype was tested using a three-page sample containing **Cytiva ÄKTA ready certificates of quality and an accompanying storage-conditions letter**. Its goal is to make answers easier to ground in the supplied material—not to guarantee that hallucinations are eliminated.

## Features

- **PDF extraction:** Extracts text from pages with PyMuPDF (`fitz`) and retains page metadata.
- **Overlapping chunking:** Splits text with LlamaIndex `SentenceSplitter` (300-token chunks, 50-token overlap).
- **Semantic embeddings:** Uses Hugging Face `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector indexing:** Indexes chunks with LlamaIndex `VectorStoreIndex`.
- **Hybrid retrieval:** Combines vector similarity search with BM25 keyword retrieval and removes duplicate nodes.
- **Cross-encoder reranking:** Uses `cross-encoder/ms-marco-MiniLM-L6-v2` to select up to five passages.
- **Grounded generation:** Passes retrieved context to Google Gemini to generate answers, with instructions not to invent unsupported information.

## RAG workflow

```text
Pharmaceutical PDF
       |
       v
PyMuPDF text extraction (with page metadata)
       |
       v
Overlapping chunks (300 tokens, 50 overlap)
       |
       v
MiniLM embeddings --> VectorStoreIndex
       |                     |
       |                     v
       +--------------> Vector retrieval
       |                     |
       +--------------> BM25 keyword retrieval
                             |
                             v
                   Merge + deduplicate
                             |
                             v
                    Cross-encoder rerank
                             |
                             v
                     Top 5 passages
                             |
                             v
                   Gemini answer generation
```

## Technology stack

| Component | Technology |
| --- | --- |
| Environment | Google Colab |
| Language | Python |
| RAG framework | LlamaIndex |
| PDF parsing | PyMuPDF (`fitz`) |
| Chunking | `SentenceSplitter` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | LlamaIndex `VectorStoreIndex` |
| Keyword retrieval | BM25 (`llama-index-retrievers-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Generative model | Google Gemini (via `llama-index-llms-google-genai`) |

## Why these design choices?

**Embeddings — MiniLM:** A compact, locally runnable embedding model that supports semantic matching when a question and a document use different terminology. No separate paid embedding API is required.

**Chunking — 300 tokens with 50-token overlap:** Helps retain context between adjacent passages while keeping retrieved excerpts focused. The configuration is a starting point rather than a proven optimum.

**Retrieval — hybrid BM25 + vector search:** BM25 is useful for exact terms such as test names, SOP identifiers, and product terminology, while vector retrieval can find conceptually related passages. A cross-encoder then reranks the combined candidates for relevance.

## Running the project in Google Colab

1. Open your project notebook in [Google Colab](https://colab.research.google.com/).
2. Install the notebook dependencies:

   ```python
   %pip install llama-index
   %pip install llama-index-llms-google-genai
   %pip install llama-index-embeddings-huggingface
   %pip install llama-index-retrievers-bm25
   %pip install pymupdf
   %pip install sentence-transformers
   %pip install nest_asyncio
   ```

3. Generate a Gemini API key in [Google AI Studio](https://aistudio.google.com/), then save it in **Colab Secrets** as `GEMINI_API_KEY`. Enable notebook access. **Do not paste keys into cells or commit them to GitHub.**
4. Run your notebook's model-configuration cells.
5. Upload the sample pharmaceutical PDF when the file-upload cell runs.
6. Run the PDF extraction, chunking, embedding/indexing, hybrid retrieval, reranking, and response-generation cells in order.
7. Execute the two evaluation questions shown below.

> **Note:** Colab runtimes are temporary. If a session resets, reinstall the dependencies and rerun the notebook cells to rebuild the index.

## Example queries and observed results

These are responses observed from the prototype during testing; they are not guarantees of identical outputs on later runs.

### Query 1: Quality-control testing

**Question:** `What test methods were used for quality control?`

The pipeline identified:

- **Flow Rate Test:** Per SOP-QC-042
- **Pressure Integrity:** Maximum 0.5 bar, 5-minute hold
- **Visual Inspection:** No defects visible
- **Package Integrity:** Sealed, no damage
- **Autoclave — Pump tubing:** 121°C for more than 15 minutes
- **Gamma Irradiation — Inlets:** 25.0–40.0 kGy

**Source:** Quality-control table in the sample PDF, page 3. Some entries specify test criteria or acceptance conditions rather than a named test procedure.

### Query 2: Storage conditions

**Question:** `What are the storage conditions specified in the certificate?`

**Observed RAG response:**

> Based on the provided document context, there are no storage conditions specified in the certificates of quality. (The document mentions storage conditions for standard and modified flow kits in the letter portion—recommending a temperature of > +5 °C—but does not specify storage conditions within the certificates themselves.)

**Source:** Storage-conditions letter, page 1 of the sample PDF. The letter recommends **storage above +5°C** for the relevant ÄKTA ready flow kits, separately from their **operating range of +2°C to +40°C**.

## What I learned

- Effective RAG depends on **document parsing, chunking, retrieval, and evidence selection**, not just the language model.
- Hybrid search can combine **exact technical terminology** with broader semantic matches.
- Reranking can help prioritize the passages most useful for answering a specific question.
- **Document boundaries matter:** the storage requirement was found in an accompanying letter, not in the certificates themselves.

## Current limitations and future improvements

- **Small demonstration dataset:** Tested on a three-page sample; retrieval quality on larger collections is not yet evaluated.
- **Score comparability:** The initial prototype merges vector and BM25 results and sorts their raw scores, which are not inherently on the same scale. A next step is **reciprocal rank fusion (RRF)**.
- **No automated answer-quality benchmark:** Future work could evaluate retrieval recall, source faithfulness, and response accuracy.
- **Source citations:** Page metadata is preserved, but the generated responses do not yet consistently print source citations.
- **Scanned PDFs:** The implemented parsing uses PyMuPDF text extraction; a Tesseract OCR fallback could be added for image-only pages.
- **Storage:** The current prototype uses an in-memory vector index; persistent storage (for example, ChromaDB) is a potential enhancement.

## Responsible use

This is a learning and portfolio project, **not a validated pharmaceutical compliance tool**. Important answers should always be verified against the original document. Only upload or publish sample documents that you have permission to share; keep API keys and sensitive materials private.

## Acknowledgment

Built during the **Pfizer × Extern AI-Powered Document Intelligence externship**, applying lessons in LlamaIndex, embeddings, chunking, hybrid retrieval, reranking, and responsible use of generative AI.
