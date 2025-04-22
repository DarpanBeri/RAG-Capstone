# RAG-Capstone
**"When Machines Evaluate Machines: A Critical Assessment of AI Evaluators in the Context of Optimizing LLaMa-Based RAG Pipelines"**

This repository shares the full Jupyter notebook and supporting materials for my capstone project at Trent University, where I systematically explored how chunk size and top_k hyperparameters affect a Retrieval-Augmented Generation (RAG) pipeline’s accuracy, and assessed the reliability of using LLMs as automated evaluators.

## 📂 Repository Structure
```
RAG-Capstone/
│
├── notebooks/             # Jupyter notebooks (analysis & demos)
│   └── capstone.ipynb     # Complete experiment pipeline & results
│
├── .gitignore             # Ignore caches, checkpoints, virtual environments
├── LICENSE                # Open‑source license (MIT)
└── README.md              # Project overview and instructions
```
## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- A Hugging Face account with access to `meta-llama/Llama-3.1-8B` (request from Meta)
- Sufficient GPU memory (Tesla T4 or equivalent) for 8‑bit quantized inference

### Installation

1. Clone this repository:
    ```
   git clone https://github.com/DarpanBeri/RAG-Capstone.git
   cd RAG-Capstone
    ```

3. (Optional) Create and activate a virtual environment:
    ```
   python3 -m venv .venv
   source .venv/bin/activate
    ```

### Running the Notebook

- Launch JupyterLab or Jupyter Notebook:
    ```
    jupyter lab # or jupyter notebook
    ```
- Open `notebooks/capstone.ipynb` and run all cells in order.

## 📖 Notebook Contents

1. Environment Setup: installs packages, configures GPU, and loads the RAG mini-wikipedia dataset.
2. Preprocessing & Chunking: text cleaning (ftfy), paragraph splitting, and sentence-based chunking functions.
3. Document Store Preparation: builds a Haystack InMemoryDocumentStore with chunk embeddings.
4. Model Handler: ModelHandler class for quantized LLaMa model loading and generation.
5. Answer Generation: generate_answers function loops through questions, retrieves context, and generates answers.
6. Automated Evaluation: evaluate_answers prompts an evaluator Llama to compare generated vs. ground truth answers.
7. Experiment Loops: single-run (run_experiment) and grid-search (run_multiple_experiments) functions, saving timestamped CSV results.
8. Results Saving: output CSVs stored alongside the notebook for downstream analysis.

## 📝 Key Findings

- Optimal Configuration: smaller chunk sizes (50–100 words) and top_k=1 yielded the highest human-rated accuracy (~70%).
- Evaluator Reliability: the LLaMa-based evaluator agreed with human judgments about 73% of the time (Cohen’s κ = 0.47), highlighting the need for human oversight.
- Hyperparameter Effects: increasing chunk_size to 200 or raising top_k decreased answer quality and evaluator agreement.

> For detailed figures, tables, and statistical analysis, please refer to the report in the resources section of my [website](https://darpanberi.github.io/)

## 📈 Viewing Results

After running experiments, CSV files appear in your working directory, named like:
`results_chunk100_top1_20250422-171305.csv`

Use pandas or Excel to explore:
```
import pandas as pd
df = pd.read_csv('results_chunk100_top1_20250422-171305.csv')
df.head()
```

## Contact & Attribution

If you find this work useful or would like to collaborate, please ⭐ the repo and reach out at:
**darpanberi (dot) 99 (at) gmail (dot) com**

I’m actively seeking opportunities in data science, AI engineering and software engineering. Let’s connect!

## License

I’ve chosen the MIT License for maximum openness and attribution. Under MIT, you’re free to use, modify, and distribute this code, provided you include the original copyright.

In return, I’d appreciate attribution (a star on GitHub!) and, if my work adds value to your projects, the opportunity to discuss potential collaborations or roles.

---

© 2025 Darpan Beri
