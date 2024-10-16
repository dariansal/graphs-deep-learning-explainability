## Introduction
This repository contains research on graph deep learning models and model explainability, focusing on Graph Convolutional Networks (GCN), Graph Isomorphism Networks (GIN), and their applications to graph classification tasks. It also includes custom implementations of datasets, fine-tuning techniques, and the use of PGExplainer for model interpretability.

## Purpose
This repository includes the following:

- Custom molecular graph classes with underlying matrix and dictionary representations
- A creation of a custom BA2MOTIF dataset from scratch
- Various GCN and GIN classification models
- Fine-tuning with edge perturbation and node feature masking
- Use of PGExplainer to explain GCN classifications on BA2MOTIF dataset by identifying explanatory subgraphs (motifs)

Additional researched areas include:
- New explainers for graph models (e.g., GNNExplainer, ProxyExplainer)
- Discrete denoise diffusion modeling for graph generation (e.g., Digress, RePaint)

## Reproducibility

1. Install Python 3.10.14.
2. Set up and activate a virtual environment:
    ```zsh
    python3.10 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install dependencies:
    ```zsh
    pip install -r requirements.txt
    ```

## Project Structure
The project is organized as follows:

- **`data/`**: 
  - **`BA2MOTIF/`**: Premade BA2MOTIF dataset
  - `custom-ba2motif.pt`: Serialized version of the custom BA2MOTIF dataset.
  - **`MUTAG/`**: Official MUTAG dataset

- **`model-reports/`**: Methodology and analysis reports
  - **`visuals/`** contains images used in the reports 
  - `BA2-PGExplainer-methodology-report.md`
  - `MUTAG-methodology-report.md`

- **`models/`**:  
  - Naming convention `{GCN/GIN}-{test-accuracy}.pth`; placeholders enclosed in {}
  - **`BA2MOTIF/`**: Trained GNNs for custom BA2MOTIF dataset
  - **`MUTAG/`**: Trained GNNs for official MUTAG dataset
  - **`PGExplainer/`**: Trained PGExplainer for BA2MOTIF classification models
  
- **`notebooks/`**: Model development process
  - `BA2MOTIF-scratch.ipynb`: custom BA2MOTIF dataset creation and analysis, GCN/GIN development, fine-tuning
  - `MUTAG-scratch.ipynb`: Custom molecular graph classes and GIN classifier
  - `PGExplainer.ipynb`: Training and inference of explainer with custom BA2MOTIF dataset

- **`PGExplain/`**: Modified PGExplainer code

- `notes.pdf`: Handwritten notes on project concepts. Listed below are the most important sections to guide understanding.
  - Table of Contents (page 1)
  - GNNs (pages 7–16)
  - PGExplainer (pages 23–34)

## Summary of Results
For detailed methodologies and analyses, refer to the reports in the **`model-reports/`**. However, for a quick summary:

- **MUTAG Classification:**
  - GIN Test Accuracy: 86.84%

- **Custom BA2MOTIF Dataset:**
  - GCN Test Accuracy: 100.00%
  - GIN Test Accuracy: 99.50% 
  - Fine-tuned GIN Test Accuracy: 100.00%

- **PGExplainer:**
  - Overall test accuracy has not yet been evaluated, but based on manual classification, it is expected to exceed 90%

## Future Plans
- Evaluate PGExplainer train and test accuracy for the entire custom BA2MOTIF dataset
- Research different explainability models to address out-of-distribution (OOD) problems (i.e., explainable subgraph is OOD when fed back into GNN) for explainer training
- Implement discrete denoise diffusion modeling for in-distribution graph generation of BA2MOTIF dataset


## Citations
This project significantly modified the code of PGExplainer for better readability and functionality. Additionally, the code was made compatible with the custom BA2MOTIF dataset.

**Bibtex citation**
```
@inproceedings{holdijk2021re,
  title={[Re] Parameterized Explainer for Graph Neural Network},
  author={Holdijk, Lars and Boon, Maarten and Henckens, Stijn and de Jong, Lysander},
  booktitle={ML Reproducibility Challenge 2020},
  year={2021}
}
```

## Usage Rights
This repository is publicly visible for academic purposes only. Any unauthorized use, reproduction, or distribution requires explicit permission from the repository owner.