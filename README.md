# RelevantCauseCAS

This is the repository for the implementations of the paper [*Finding relevant causes in complex systems: A generic method adaptable to users and contexts*]([Link comming soon, paper accepted at ACSOS 2025](https://ieeexplore.ieee.org/abstract/document/11217769)).

## About this project

Complex Adaptive Systems (CAS) feature intricate dynamics that are difficult to track and control. This paper addresses the challenge of providing causal explanations for unwanted or unexpected events in CAS, e.g. “why did this happen?”. Previous work in cognitive science indicates that satisfactory explanations should identify relevant causes tailored to the user's query and context. While automated methods exist for generating causal explanations, they typically overlook relevance or rely on a static notion of “best cause”. We propose a novel two-step methodology to provide relevant causes: (1) identify a wide range of causes; (2) rank and filter them using a flexible evaluation framework that can be tailored to users. We rely on an existing cause-detection method (1); and focus on selecting the most relevant ones (2). We propose three relevance metrics and a flexible method for combining them. We validate our framework using a flocking simulation where agents encounter obstacles. Experiments demonstrate how the flexibility of our method allows us to generate diverse causal explanations, each highlighting the most relevant aspects to the user.

## How to use

To use this code, you can clone the repository. You can setup an environment for jupyter notbook using:

```bash
    source setup.sh
```

The code for the experiments can be found in the [src](src/) folder. You can find examples of usage or render the result figures in [this notebook](src/main.ipynb). 

To reproduce the experiment from the paper, run:

```bash
    python src/experiments.py
```

## Citation

If you use this code for an academic work, please use the following citation:

Text:
> S. Reyd, A. Diaconescu and J. -L. Dessalles, "Finding Relevant Causes in Complex Systems: a Generic Method Adaptable to Users and Contexts," 2025 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS), Tokyo, Japan, 2025, pp. 100-111, doi: 10.1109/ACSOS66086.2025.00026.

BibTex:
> @INPROCEEDINGS{11217769,author={Reyd, Samuel and Diaconescu, Ada and Dessalles, Jean-Louis},booktitle={2025 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS)},title={Finding Relevant Causes in Complex Systems: a Generic Method Adaptable to Users and Contexts},year={2025},pages={100-111},doi={10.1109/ACSOS66086.2025.00026}}
