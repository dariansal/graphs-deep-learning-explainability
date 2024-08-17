
## Purpose
This repository was created to gain a better understanding of some of the research of graph deep learning models and model explainability done this summer. This includes:
- Classes to create molecular graphs with both underlying matrix representations (used by PyTorch) and dictionary representations (used by NetworkX)
- Various Graph Convolutional Network (GCN) and Graph Isomorphism Network (GIN) classification models
- A creation of a custom BA2MOTIF dataset from scratch
- Fine-tuning with edge pertubation and node feature masking
- Use of PGExplainer to explain GIN classifications on BA2MOTIF dataset by extracting motif subgraph

In addition to this code, I also researched other popular explainers (e.g., GNNExplainer, ProxyExplainer) and learned of the basics of discrete denoise diffusion modeling (e.g., Digress, RePaint) for graph generation.

## Reproducibility
To reproduce the project:
1. Ensure Python 3.10.14 is installed.
2. Set up a virtual environment using Python 3.10.14.
3. Run `pip install -r requirements.txt` to install the necessary packages.

## Project Structure
The project is organized as follows:

- **`data/`**: 
  - **`BA2MOTIF/`**: Contains premade BA2MOTIF dataset
  - `custom-ba2motif.pkl`: Serialized version of the custom BA2MOTIF dataset.
  - **`MUTAG/`**: Contains the official MUTAG dataset

- **`model-reports/`**: Contains reports of the methodologies and analysis of each of the models
  - **`visuals/`** contains images that were present in the reports 
  - `MUTAG-model-reports.md` 
  - `BA2MOTIF-model-report.md`

- **`models/`**: Contains saved graph classification GNNs:
  - The models contain the following naming convention (`{GCN/GIN}-{test-accuracy}.pth`, placeholders enclosed in {})
  - **`BA2MOTIF/`**: Contains GNNs for custom BA2MOTIF dataset
  - **`MUTAG/`**: Contains GNNs for official MUTAG dataset
  
- **`notebooks/`**: Contains the model development process for each of the models created
  - `BA2Motif-scratch.ipynb` contains analysis of BA2MOTIF dataset, custom implementation of dataset, development of GCN and GIN, fine-tuning, and addition of PGExplainer
  - `MUTAG-scratch.ipynb`: Includes creation of molecular graph classes with underlying matrix and dictionary structures. Also includes GIN classifier

- `notes.pdf`: Handwritten notes created to better understand and complete the project. Listed below are the most important sections to guide understanding.
  - Table of Contents (page 1)
  - GNNs (pages 7–16)
  - PGExplainer (pages 23–34)


## Citations
I modified the code (cited below) for PGExplainer, a parameterized explainer, to be compatible with the custom BA2MOTIF dataset I created. 

**Bibtex citation**
```
@inproceedings{holdijk2021re,
  title={[Re] Parameterized Explainer for Graph Neural Network},
  author={Holdijk, Lars and Boon, Maarten and Henckens, Stijn and de Jong, Lysander},
  booktitle={ML Reproducibility Challenge 2020},
  year={2021}
}
```
## Summary of Results
Details of the models and their methodologies provided in **`model-reports/`**. However, for a quick summary, the MUTAG classification GIN had a test accuracy of 86.84%. Additionally, for the custom BA2MOTIF dataset, the test accuracies for the GCN, GIN, fine-tuned GIN, were 100%, 99.50%, and 100% respectively.

## Future Plans
- [In Progress] Evaluate PGExplainer for the entire dataset
- Implement PGExplainer from scratch
- Research different graph explainability models and approaches to solve the out-of-distribution (OOD) problems for training explainability models (i.e., explainable subgraphs are OOD when fed back into graph classification models)
- Add a form of discrete denoise diffusion modeling for graph generation


