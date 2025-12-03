# Concept-Guided Interpretability via Neural Chunking
This repository contains code and resources to replicate the experiments from the paper


**Wu, Shuchen; Alaniz, Stephan; Karthik, Shyamgopal; Dayan, Peter; Schulz, Eric; Akata, Zeynep.**  
*Concept-Guided Interpretability via Neural Chunking.*  
Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS), 2025.  

The repository provides implementations of the chunking-based interpretability methods introduced in the paper, along with example configurations, training scripts, and evaluation routines. We hope this serves as a useful resource for exploring neural chunking as a path toward concept-guided interpretability in neural networks.
Please don't hesitate to contact me at shuchen.wu at alleninstitute.org or open an issue in case there is any question!


## 📂 Repository Structure  
```
├── RNN_experiments/ # Replication of RNN experiments
│ ├── experiment_data/ # folder to store experimental data
│ ├── RNN_1_reflection.ipynb # simple reflection hypothesis experiments on RNN
│ ├── RNN_2_artificially_induce_compositionality.ipynb # grafting RNN to induce compositionality
│ ├── RNN_3_context_dependent_chunks.ipynb # measure neural chunks and their dependency on context
│ ├── RNN_4_hierarchy.ipynb # study the number of internal chunks and the level of hierarchy in sequence 
├── data_collection/ # code to collect LLM activation data 
├── PA/ # code to replicate population averaging
├── UCD/ # code to discover chunks in an unsupervised way
├── requirements.txt # Python dependencies
├── README.md # Project overview (this file)
```
## 🚀 Getting Started  

1. Clone this repository:  
   ```bash
   git clone https://github.com/swu32/Chunk-Interpretability.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt


---

## 📖 Citation
If you use this code in your research, please cite:
```
@inproceedings{wu2025concept,
  title     = {Concept-Guided Interpretability via Neural Chunking},
  author    = {Wu, Shuchen and Alaniz, Stephan and Karthik, Shyamgopal and Dayan, Peter and Schulz, Eric and Akata, Zeynep},
  booktitle = {Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=o87dDXYLXC}
}
```

APA:
Wu, S., Alaniz, S., Karthik, S., Dayan, P., Schulz, E., & Akata, Z. (2025).
Concept-guided interpretability via neural chunking.
In Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS).
Retrieved from https://openreview.net/forum?id=o87dDXYLXC
