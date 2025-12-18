# arXiv Domain Expert Research Assistant
A research-grade question answering system for arXiv papers that retrieves, ranks, and synthesizes domain-specific scientific knowledge using advanced NLP (BERT), sentence-level extraction, and LLM reasoning — without preloading the entire dataset.

Designed for deep technical questions in domains such as Astrophysics, Condensed Matter Physics, Mathematics, and High-Energy Physics.

## Key Features
* Query-time retrieval from arXiv metadata (no full dataset loading)
* Domain-aware reasoning (e.g., astro-ph, cond-mat, hep-th)
* Multi-stage semantic ranking
    - Lexical filtering
    - Bi-encoder semantic similarity
    - Cross-encoder deep relevance scoring
    - Sentence-level information extraction
    - Citation-grounded answers (arXiv references only)
    - Hallucination-safe prompting
* Streamlit-based interactive UI


## Extract Dataset from Kaggle
```bash
kaggle export API_KEY = "your kaggle api key here"
kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip
```

## Run the application
```bash
cd task_4
streamlit run main_4.py
```

## Example Queries
* "How do the spectroscopic results distinguish between the King & Wynn (1999) stream-fed model and the Norton et al. (2004) ring-fed accretion model for EX Hya?"
* (Follow up) "Which specific Doppler tomogram features favour ring-fed accretion over direct stream accretion?"
* (Follow up) "Why is the presence of a bright spot still compatible with a ring-fed model?"


## Recorded Video
```
https://drive.google.com/file/d/1gvdCBfQoeyCs9Hougg1llOSUhco0dHbF/view?usp=sharing
```