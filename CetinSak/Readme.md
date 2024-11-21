# How Deep Networks Learn Sparse and Hierarchical Data: the Sparse Random Hierarchy Model

This readme file is an outcome of the [CENG501 (Fall 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Fall 2024) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

Deep Learning has become a foundational foot of the modern machine learning, exhibiting well performance across a wide range of problems.This success is often be evaluated to its ability to build hierarchical representations, progressing from simple features to more complex ones. In addition,  the ability to learn invariance to task-specific transformations, such as spatial changes in image data, has been strongly correlated to this performance Still, there is a fundamental question needs to be answered  **_What underlying properties make high-dimensional data effectively learnable by deep networks?_** 

This work introduces the Sparse Random Hierarchy Model (SRHM), demonstrating that sparsity in hierarchical data enables networks to learn invariances to such transformations.  Taking the RHM as a framework, this work introduces the Sparse Random Hierarchy Model (SRHM), demonstrating that sparsity in hierarchical data enables networks to learn invariances to such transformations, published at Proceedings of Machine Learning Research. It was shown as a spotlight poster at ICML 2024.

Our main goal is to create the dataset by using SRHM and reproducing the same results shared in this paper. Through systematically creation of the dataset, we aim to validate that deep networks can learn such invariances from the data in which size is  polynominal in the input dimension, emphasising their advantage over shallow networks. Furthermore, we want to verify the theoretical relationships between sparsity, sample complexity and hierarchical representations.  

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

- Summary

- Method -> SHRM, 
- Contributions
- Existing Literature -> Random Hierarchy Model, 

# 2. The method and our interpretation

## 2.1. The original method

-> SHRM, Sample Complexity, Sparsity ve Diffeomorphism, Derivation of Sample complexity for CNN and LCN (CNN but no weight sharing)

@TODO: Explain the original method.

## 2.2. Our interpretation

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

-> Interpret the output of SHRM, how should data look etc.
-> Interpret how S_k and D_k was calculated. 
-> In RHM code, what does each parameter correspond to here

# 3. Experiments and results

## 3.1. Experimental setup

1. Making SHRM dataset -> main part of the paper, explain the code in relation to the 'our interpretation' part
2. Running CNN and LCN on them -> hyperparams come from appendix
3. Getting graphs from results
4. Running CIFAR10 with given architectures -> hyperparams come from appendix, nets come from citation
5. Getting graphs from here
6. Additional guidance on how to run Appendices


@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

1. Explain the directory -> where what is stored -> nets in one folder, RHM and SHRM in another folder, main files in root, guided ipynb
2. TODO -> Add a helper script to download CIFAR dataset. Required for first milestone.
3. How to run -> venv creation, running files, reproducing images, requirements,

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

Main Paper: `How Deep Networks Learn Sparse and Hierarchical Data: the Sparse Random Hierarchy Model`
```bibtex
@InProceedings{pmlr-v235-tomasini24a,
  title = 	 {How Deep Networks Learn Sparse and Hierarchical Data: the Sparse Random Hierarchy Model},
  author =       {Tomasini, Umberto Maria and Wyart, Matthieu},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {48369--48389},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/tomasini24a/tomasini24a.pdf},
}
```

Random Hierarchy Model Base Implementation From: `How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model`
```bibtex
@article{Cagnetta_2024,
   title={How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model},
   volume={14},
   ISSN={2160-3308},
   url={http://dx.doi.org/10.1103/PhysRevX.14.031001},
   DOI={10.1103/physrevx.14.031001},
   number={3},
   journal={Physical Review X},
   publisher={American Physical Society (APS)},
   author={Cagnetta, Francesco and Petrini, Leonardo and Tomasini, Umberto M. and Favero, Alessandro and Wyart, Matthieu},
   year={2024},
   month=jul }

```

CIFAR 10 Dataset From: `Learning Multiple Layers of Features from Tiny Images`
```bibtex
@inproceedings{Krizhevsky2009LearningML,
  title={Learning Multiple Layers of Features from Tiny Images},
  author={Alex Krizhevsky},
  year={2009},
  url={https://api.semanticscholar.org/CorpusID:18268744}
}
```

Model Implementations (VGG ResNet EfficientNetB0) are from: [diffeo-sota repository](https://github.com/leonardopetrini/diffeo-sota/tree/main/models)

# Contact

Burak Erinç Çetin - erinc.cetin@metu.edu.tr
Emin Sak - sak.emin@metu.edu.tr
