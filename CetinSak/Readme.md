# How Deep Networks Learn Sparse and Hierarchical Data: the Sparse Random Hierarchy Model

This readme file is an outcome of the [CENG501 (Fall 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Fall 2024) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

## 1. Introduction

Deep Learning has become a foundational foot of the modern machine learning, exhibiting well performance across a wide range of problems.This success is often be evaluated to its ability to build hierarchical representations, progressing from simple features to more complex ones. In addition, the ability to learn invariance to task-specific transformations, such as spatial changes in image data, has been strongly correlated to this performance Still, there is a fundamental question needs to be answered **_What underlying properties make high-dimensional data effectively learnable by deep networks?_**

This work introduces the Sparse Random Hierarchy Model (SRHM), demonstrating that sparsity in hierarchical data enables networks to learn invariances to such transformations. Taking the RHM as a framework, this work introduces the Sparse Random Hierarchy Model (SRHM), demonstrating that sparsity in hierarchical data enables networks to learn invariances to such transformations, published at Proceedings of Machine Learning Research. It was shown as a spotlight poster at ICML 2024.

Our main goal is to create the dataset by using SRHM and reproducing the same results shared in this paper. Through systematically creation of the dataset, we aim to validate that deep networks can learn such invariances from the data that is polynominal in the input dimension, emphasising their advantage over shallow networks. Furthermore, we want to verify the theoretical relationships between sparsity, sample complexity and hierarchical representations.

### 1.1. Paper summary

Deep learning can solve high dimensional tasks. This is possible because learnable data is highly structured. Data is learnable when it has local features that are assembled hierarchically. This view is consistent with deep networks forming hierarchical representations or CNNs architectural choices. Hierarchical nature of data can be captured by using hierarchical models.

However, hierarchical models are discrete, whereas images are approximated as continuous functions. Labels on this view are invariant to smooth transformations. Enforcing stability to smooth transformations (diffeomorphisms) makes models generalize better. Thus, there is a strong correlation between a network's test error and its sensitivity to diffeomorphisms.

Authors argue that:

- Incorporating sparsity to hierarchical generative models leads to insensitivity to discrete versions of diffeomorphisms.
- To illustrate, they introduce the Sparse Random Hierarchy Model (SRHM), which captures the empirically observed correlation between sensitivity to diffeomorphisms and test error.
- Correlation between test error reduction and invariance to diffeomorphisms occur when a model has learnt a hierarchical representation.
- Number of training points needed to learn the task, called sample complexity, is the point where both diffeomorphism insensitivity and low test error is achieved.

### Prior Work

Deep networks can represent hierarchical compositional functions with less parameters compared to shallow networks. Specifically, deep networks can learn a hierarchical model polynomial in the dataset dimension. This work focuses on sparsity in feature space, which corresponds to smooth transformations of the input.

Sample complexity, number of training examples needed to learn the task is introduced by the authors in their previous paper: “How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model”. It depends on the nature of the dataset, and the learning networks architecture. For example, data points required by a deep network to learn a task is usually polynomial in data dimension, but it is exponential for a shallow network (Cagnetta et al., 2024).

### Aims

- Show and quantify strong correlation between feature sparsity and insensitivity to discretized diffeomorphisms
- Explain how invariance emerges during training and how it affects sample complexity.
- Work with generic perturbations of data, rather than adversarial-like worst-case perturbations.

## 2. The method and our interpretation

### 2.1. The original method

### Random Hierarchical Model (RHM)

The Random Hieararchy Model is a generative model that illustrates how hierarchical composition of features can be used by deep neural networks to a specific tasks. It is inspired by the structure of natural data like images and language, where features are composed hierarchically. In this model:

- **_Hierarchy of Features_**: The way of the data being It starts with the class labels at the top, which are like the “parents.” Each class then leads to a group of high-level features, these represent general characteristics like "face" or "background" in an image. These high-level features then break down further into smaller parts, like "eyes" or "nose." This process keeps going until we reach the very bottom, where we get the simplest building blocks, like edges or pixels in an image. It’s a bit like how real-world objects are made up of smaller, simpler pieces that come together to form something complex.
- **Randomness**: The generation rules for features at each level are chosen randomly, ensuring a wide variety of possible feature compositions.

Deep networks, then learn the "RHM task" by developing representations that are invariant to interchange of the synonymic features.

A core point of the RHM is its sample complexity $`P^*`$, the number of training points that is required to generalize the model to be trained. The $`P^*`$ is corraleted with:

- $`n_c`$: Number of classes.
- $`m`$: Number of equivalent sub-representations for each feature.
- $`L`$: Depth of the hierarchy.

The paper frames the sample complexity in two key questions:

- "_**Why are shallow networks cursed?**_": Shallow networks, such as two-layer networks, are unable to exploit the hierarchical structure of data. To learn tasks modeled by the RHM, these networks needs huge amount of data. This results in an exponential sample complexity $`P^* \sim n_c m^{\frac{d-1}{s-1}}`$ where $`d`$ is the input dimension. As $`d`$ grows, the required training data becomes huge, making it impractical for shallow networks to generalize effectively.

- "_**How do deep networks break the curse?**_": Deep networks are a great fit for hierarchical tasks because they follow the same structure as the RHM. They learn step by step, building representations at each level of the hierarchy. This approach simplifies the task by reducing its overall complexity. Thanks to this, their sample complexity is $`P^* \propto n_c m^L`$($`\frac{P^*}{n_c}≃d^{\frac{ln(m)}{ln(s)}}`$) which grows only polynomially with $`d`$. This makes deep networks capable of handling tasks effectively.

### Sparse Random Hierarchical Model (SRHM)

The **Sparse Random Hierarchical Model (SRHM)** extends the RHM by introducing **sparsity**, where a majority of features are "uninformative" (e.g., empty or irrelevant) while only a few features contain useful information for classification. This sparsity mimics real-world data, such as images where only a subset of pixels is relevant to the label.

Before diving into the generation process, we define the key symbols and terms:

- $`C = \{1, \ldots, n_c\}`$: The set of class labels, where \( n_c \) is the number of classes.
- $`V_\ell`$: The vocabulary at level $`\ell`$, containing possible features at that level.
- $`\mu_i^{(\ell)}`$: A specific feature at level $`\ell`$ within the vocabulary $`V_\ell`$.
- $`s_\ell`$: The number of lower-level features generated by each higher-level feature at level $`\ell`$.
- $`L`$: The total depth of the hierarchy, with level $`\ell = L`$ being the highest and $`\ell = 1`$ being the lowest.

The data generation can be summarized as:

1. **Class Label to High-Level Features**
   At the top of the hierarchy, a class label $`\alpha\in C=\{1, \ldots,n_c\}`$ generates $`s_L`$ high-level features $`(\mu_i^{(L)} \in V_L)`$, using rules: $`\alpha\to \mu_1^{(L)}, \ldots, \mu_{s_L}^{(L)}.`$
2. **High-Level to Low-Level Features**
   Each feature $`\mu_i^{(\ell)} \in V_\ell`$ at level $`\ell`$ generates $`s_\ell`$ lower-level features $`\mu_1^{(\ell-1)}`$ $`, \ldots, \mu_{s_\ell}^{(\ell-1)},`$, using rules $`\mu_i^{(\ell)} \to \mu_1^{(\ell-1)}, \ldots, \mu_{s_\ell}^{(\ell-1)}, \quad \text{for } \ell = 2, \ldots, L`$.
3. **Low-Level Features to Input**
   At the lowest level $`(\ell = 1)`$ the features $`\mu_i^{(1)} \in V_1`$ correspond to the input dimensions, such as pixels in an image.

#### Sparsity and Diffeomorphism

Sparsity in SRHM makes models robust to spatial transformations.This sparsity is introduced by adding **uninformative features** to each patch of data simulating diffeomorphisms. Informative features are embedded within these uninformative regions, making the data sparse and reducing the effective dimensionality of the informative components.
![Figure-2 From Paper](assets/figure2.png)

They register this sparsity in two ways:

- **Sparsity A (Fixed Positions):**

  - Each of the $`s`$ informative features is embedded in a fixed position within a sub-patch of size $`s_0+1`$, where $`s_0`$ denotes the number of uninformative features.
  - Example From _Figure-2_:
    - In a patch, $`s=2`$ informative features are surrounded by $`s_0=1`$ uninformative feature, forming a sub-patch of size $`s(s_0+1)=6`$.
    - Informative features remain in fixed positions across the hierarchy. For instance, the patch might always have "red" and "blue" in fixed spots, while "empty" spaces represent uninformative features.

- **Sparsity B (Flexible Positions):**
  - The $`s`$ informative features can occupy any position within the sub-patch of size $`s(s_0+1)`$, but their order remains consistent.
  - Example From _Figure-2_:
    - The two informative features ("red" and "blue") can move around within the patch, as long as their sequence is preserved. The uninformative features can appear anywhere else in the patch.

Sparsity propagates through the hierarchical generation process:

- At each level, only the $`s`$ informative features produce meaningful descendants, while the $`s_0`$ uninformative features produce further uninformative patches in the next level.
- This propagation creates a sparse representation where only a small subset of all input features contributes to the task.

#### Sample Complexity

The SRHM distinguishes itself from the RHM by incorporating a sparsity factor, impacts sample complexity. They find that sparsity factor influences the sample complexity in CNN and LCN in a different way while the keeping the synonmys $`m = v^{s-1}`$ as the $`s, s0, v`$ and $`L`$ are changed

- **LCN**
  For a LCN, each filter learns independent weights. The total number of parameters is proportional to the input dimension $`d = \left( s (s_0 + 1) \right)^L`$ and the number of filters at each level. Also The sparsity factor $`(s0​+1)`$ increases the number of uninformative features, making the task harder. The effective complexity grows exponentially with $`L`$ and the sparsity factor. By knowing these and from the experimental results, they found that sample complexity $`P^*_{LCN} \sim C_0 (s_0 + 1)^L n_c m^L`$($`C_0`$ is a constant dependent on the architecture and training conditions.).
- **CNN**
  For a CNN, The number of parameters is proportional to the filter size $`s(s_0 + 1)`$ and does not scale with the input size $`d`$. This reduction in parameters makes CNNs more efficient than LCNs.While sparsity still affects the effective input size, weight sharing reduces its impact. The sample complexity scales quadratically with $`(s_0+1)`$ instead of exponentially. By knowing these and from the experimental results, they found that sample complexity $`P^*_{CNN} \sim C_1 (s_0 + 1)^2 n_c m^L`$($`C_1`$ is a constant dependent on the architecture and training conditions.).

### 2.2. Our interpretation

The output of the Sparse Random Hierarchical Model (SRHM) is a **set of generated hierarchical data points** that follow a structured and sparse pattern. Specifically:

- _**Data**_:Each data point consists of a sparse input $`( x \in \mathbb{R}^{d \times v} )`$, where:

  - $`( d = (s(s_0 + 1))^L )`$: The input dimension, representing the number of sub-features across all levels of the hierarchy.
  - $`v`$: The one-hot encoded(or many other encoding techniques) vocabulary size for each feature.

The input $`x`$ represents a hierarchical composition of informative features (useful for classification) and uninformative features (noise or placeholders).

- _**Label**_: Each data point also has a corresponding class label $`y \in C = \{1, 2, \ldots, n_c\},`$

The label $`y`$ is derived from the top-level (class-level) feature, which propagates down through the hierarchy to produce $`x`$.

Paper introduces $`S_k`$(sensitivity to synonymic exchanges) and $`D_k`$ (sensitivity to diffeomorphisms) as metrics to evalute success, how well a network learns invariances to transformations and exchanges.

_**$`S_k`$ (Sensitivity to Synonymic Exchanges)**_

$`S_k`$​ measures how sensitive the activations of the $`k`$-th layer of the network are to synonymic exchanges at a  hierarchical level $`\ell`$.  This formula given as

$`S_k = \frac{\langle \| f_k(x) - f_k(P_\ell x) \|_2 \rangle_{x, P_\ell}}{\langle \| f_k(x) - f_k(y) \|_2 \rangle_{x, y}}`$ where:

- $`f_k(x)`$:Activations of the $`k`$-th layer for input $`x`$.
- $`P_\ell`$:A transformation operator that exchanges level-$`\ell`$ synonyms.
- $`( \| \cdot \|_2 )`$:Euclidean norm.
- $`( \langle \cdot \rangle_{x, P_\ell} )`$:Average over inputs $`x`$ and synonym transformations $`( P_\ell )`$.
- $`( \langle \cdot \rangle_{x, y})`$: Average over pairs of unrelated inputs $`x`$ and $`y`$.

This process can be summarized as:

- The numerator computes the average change in layer $`k`$'s activations when synonyms at level $`\ell`$ are swapped in the input.
- The denominator normalizes this by the average difference in activations between unrelated inputs.

_**$`D_k`$ (Sensitivity to Diffeomorphisms)**_

$`D_k`$​ measures how sensitive the $`k`$-th layer is to diffeomorphic transformations, such as translations or small deformations, applied to the input. Calculating this kind of sensitivities as a metric requires below formula:

$`D_k = \frac{\langle \| f_k(x) - f_k(\tau(x)) \|_2 \rangle_{x, \tau}}{\langle \| f_k(x) \|_2 \rangle_x}`$ where

- $`f_k(x)`$:Activations of the $`k`$-th layer for input $`x`$.
- $`\tau(x)`$:Input $`x`$ after applying a diffeomorphic transformation $`\tau`$.
- $`( \| \cdot \|_2 )`$:Euclidean norm.
- $`( \langle \cdot \rangle_{x, \tau} )`$:Average over inputs $`x`$ and synonym transformations $`( P_\ell )`$.
- $`( \langle \cdot \rangle_{x})`$: Average over inputs $`x`$

This process can be summarrized as:

- The numerator computes the average change in activations when the input is slightly transformed.
- The denominator normalizes this by the average magnitude of the activations.

  **Low $`S_k`$​ and $`D_k`$:**

  - Indicates that the network has successfully learned invariant representations.
  - For example, if $`S_k`$ is low, the network is insensitive to swapping synonyms, meaning it focuses on the overall structure rather than the specific representation of features.
  - Similarly, low $`D_k`$ shows robustness to small spatial transformations, such as shifts or deformations.

   **High $`S_k`$​ and $`D_k`$**

- Indicates sensitivity to synonymic exchanges or transformations, meaning the network hasn’t fully generalized to the invariances of the task.

The RHM and the SHRM model parameters meaning nearly the same the only difference is coming from the registered uninformative results to the features. Meaning that:

- The total inut dimension is modified from $`d=s^L`$ to $`d = (s(s_0 +1))^L`$ where the $`s_0`$ is the sparsity factor(number of the uninformative elements in a chunk).

## 3. Experiments and results

### 3.1. Experimental setup

Explanations of each experiment is below. There are sections from 1 to 4 for the main paper and from A to G for appendices. Each section has its goal, explanation, figure and implementation parts. Each section has a corresponding experiment file stored under `src/experiments`. 

There are two ways to run our experiments:

1. Follow the experiments.ipynb notebook and run each experiment.
2. `src/main.py` runs each experiment from its file. Experiments are stored like `src/experiments/experiment_L.py` where L is the label of the experiment. 

---

### Experiment 1 - Sample Complexity of LCN and CNN

**Goal** : Find Sample Complexities of Locally Connected Networks (LCN) and CNNs.

#### Explanation of Experiment 1

- **A** : Test error versus number of training points P is plotted. To extract the sample complexity, authors defined an arbitrary threshold of reaching 10% test error. This threshold is based on this plot.
- **B** :  Empirical sample complexity for a LCN to reach 10% test error.
- **C** : Empirical sample complexity for a CNN to reach 10+ test error.

The first plot is drawn for selecting the 10% error rate. The second and third plots shows the correlation between sample complexity values of networks with reaching a 10% test error.

- For LCN empirical sample complexity is : $`(s^{L/2})(s_0+1)^Ln_cm^L`$
- For CNN empirical sample complexity is : $`(s_0+1)^2n_cm^L`$

![Figure 3](assets/figure4.png)

#### Implementation of Experiment 1

Implementation of Experiment 1 is under `src/experiments` and can be run using `python -m src.experiments.experiment_orchestrator --config_name exp1N_config.toml` where N is a, b or c.

Figures below show our results for experiment figure B and C. For figure A, we see that our plots agree with the result provided by authors. Our figures do not agree with authors figures. While training, we could not determine the exact point we crossed to 90% accuracy, thus we have noise. This is especially true for the outlier on the second figure.

![Experiment 1 A](assets/exp1a.png)
![Experiment 1 B](assets/exp1_b.png)
![Experiment 1 C](assets/exp_1c.png)

---

### Experiment 2 - Benefit of Sparsity

**Goal** : Show that sparsity in the in the dataset makes sample complexity of LCN lower.

#### Explanation of Experiment 2

Fraction of informative pixels is introduced, which is $`F = (s_0 + 1)^{−L}`$ as a measure of sparsity. Lower $F$ means lower number of informative pixels, thus higher sparsity. Reformulating $`P^*_{LCN}`$ in terms of $F$ gives:

$$
P^∗_{LCN} ∼ F^{\frac{logm}{logs} - \frac{1}{2}}d^{\frac{logm}{logs} + \frac{1}{2}}
$$

This is an increasing function as long as $m>\sqrt{s}$. When m and s is fixed and d and F varies, plot below occurs for an LCN.

![Figure 5](assets/figure5.png)

#### Implementation of Experiment 2

Plotting the given parameters by defining a linear space of points, and applying the dimension and relevant fraction solves this. Below is the graph.

![Figure 5 Implementation](assets/exp2.png)

---

### Experiment 3 - LCN Sample Complexity, Synonym and Diffeomorphism Sensitivity

**Goal**: Show that sample complexity of a LCN is equivalent to its sample complexity for synonymic sensitiviy and to its sample complexity for diffeomorphism sensitivity.

#### Explanation of Experiment 3

$`P^{*}_S ≈ P^{*}_{LCN}`$ and $`P^{*}_D ≈ P^*_{LCN}`$ where:

- $`P^*_S`$ is the sample complexity of LCN for $`S_2`$. $`S_2`$ is the synonymic sensitivity at layer 2.
- $`P^*_D`$ is the sample complexity of LCN for $`D_2`$. $`D_2`$ is the diffeomorphism sensitivity at layer 2.

Showing this means that for a LCN, the model solves the task when it also becomes resilient to synonymic and diffeomorphic differences in the dataset.

![Figure 6](assets/figure6.png)

#### Implementation of Experiment 3

Our implementations had bugs, so please see discussion part.

---

### Experiment 4 - Test Error vs Diffeomorphism Sensitivity

**Goal** : Show that correlations of diffeomorphism and synonimic sensitivity of CNN networks match with common architectures trained on CIFAR10 dataset.

#### Explanation of Experiment 4

Figures A and B are taken from [Petrini et al. 2021](https://arxiv.org/pdf/2105.02468)

Figures C and D are for common architectures trained on SRHM. Figure C is for models to reach 10% test error, compared to sensitivity to diffeomorphisms. Figure D is similar, except diffeo is changed with synonym sensitivity. Figures E and F are plots with increasing number of training examples.

![Figure 1](assets/figure1.png)
![Figure 1 Explanation](assets/figure1_explanation.png)

#### Implementation of Experiment 4

Our implementations had bugs, so please see discussion part.

---

### Experiment From Appendix A - Sparsity B

**Goal** : Show that Sparsity A and B are equivalent.

#### Explanation of Experiment A

Plotting Sparsity A and Sparsity B together shows that they are equivalent when calculating sample complexity with them.

![Figure 7](assets/figure7.png)

![Figure 8](assets/figure8.png)

---

### Experiment from Appendix B - Common Architecture Learning SRHM

**Goal** : Show common architectures learning SRHM. Vertical axis is for depth which sensitivity is measured, and horizontal axis is for different models. For each model, deeper layers are less sensitive to diffeomorphisms and synonyms.

#### Explanation of Experiment B

![Figure 9](assets/figure9.png)

#### Implementation of Experiment B

@Todo for experiment B

---

### Experiment From Appendix D - Sample Complexity of LCN

**Goal** : Empirically find the sample complexity $`P^*`$ of the LCN architecture.

#### Explanation of Experiment D

Formally show that $`P^*_{LCN} ∼ C_0(s, L)(s_0 + 1)^Ln_cm^L`$ where:

- $`C_0(s, L)`$ is proportional to $`s^{L/2}`$.
- $`s`$ is number of informative features.
- $`s_0`$ is the number of uninformative features in a patch.
- $`n_c`$ is number of classes.
- $`m`$ is the number of synonyms per feature.
- $`L`$ is the number of layers of the SRHM network.

![Figure 10](assets/figure10.png)
![Figure 11](assets/figure11.png)
![Figure 12](assets/figure12.png)

#### Implementation of Experiment D

@TODO for Experiment D

---

### Experiment From Appendix E - Sample Complexity of CNN

**Goal** : Empirically find the sample complexity $`P^*`$ of the CNN architecture.

#### Explanation of Experiment E

Formally show that $`P^*_{CNN} ∼ C_1(s_0 + 1)^2n_cm^L`$ where:

- $`C_1`$ is a constant.
- $`s_0`$ is the number of uninformative features in a patch.
- $`n_c`$ is number of classes.
- $`m`$ is the number of synonyms per feature.
- $`L`$ is the number of layers of the SRHM network.

![Figure 13](assets/figure13.png)
![Figure 14](assets/figure14.png)
![Figure 15](assets/figure15.png)

#### Implementation of Experiment E

@TODO impelemntation of experiment E

---

### Experiment From Appendix F - Sample Complexity of FCN

**Goal** : Empirically find the sample complexity of a FCN (Fully Connected Network).

#### Explanation of Experiment F

In figure at the left, plot sample complexity to reach 10% test error vs sample complexity to reach 30% $`S_2`$ synonimic sensitivity. In figure at the right, it is the same expect x-axis is for sample complexity to reach 10% $`D_2`$ diffeomorphism sensitivity.

![Figure 16](assets/figure16.png)

#### Implementation of Experiment F

@TODO experiment f

---

### Experiment From Appendix G - $`S_k`$ and $`D_k`$ Sensitivity To Permutations

**Goal** : Plot the sensitivity of changes to $`S_k`$ and $`D_k`$, where k is the layer number.

#### Explanation

Correlation between test error and change of the hidden representation of the model.

![Figure 17](assets/figure17.png)

#### Implementation of Experiment G

@TODO implemenetation of experiment G

---

### 3.2. Running the code

The code is structured as follows:

- `assets` folder has figures from the paper.
- `src` folder has the paper implementation. It contains:
  - `models` has implementations of resnet, vgg etc
  - `SRHM` has the papers implementation
  - `utils` has utility functions such as CIFAR10 dataloader
  - `main.py` is the code for
- `tests` folder has tests for our implementations.
- `experiments.ipynb` is an annotated version of the paper. It reflects the code from `main.py`, but it is interactive.
- `requirements.txt` has the dependencies required to run the experiments.

To run the code:

1. Make sure you have `Python3.11` installed. Some imports do not work with `Python3.12`.
2. Clone the repository or copy folder of this implementation. Make sure you are at the root directory of this project.
3. Create a virtual environment using `python -m venv .venv`.
4. Activate the venv using:
   - Linux/Mac : `source .venv/bin/activate`
   - Windows Powershell : `.venv/Scripts/activate.bat`
   - Windows cmd : `.venv/Scripts/activate.bat`
5. Install dependencies with `pip install -r requirements.txt`
6. Run the code from `experiments.ipynb` or from `src/main.py`

### 3.3. Results

| Note: Due to time constraints, experiments in the appendix were not implemented.


#### 3.3.2 Experiment 1 Results

From the results of experiment 1, we can see that our results deviate greatly from the authors results.

#### 3.3.3 Experiment 2 Results

#### 3.3.4 Experiment 3 and 4 Results

Due to bugs in the experiment 3 and 4, we were not able to implement them, this is due to:

- For experiment 3, synonym and diffeo sensitivity calculations were behaving erratically. For example, sum of difference of activations on

## 4. Conclusion

### 4.1 Coding The SRHM and Training Models

While coding the SRHM, we based our implementation from authors previous paper on Random Hierarchy Models. We concluded that basing the SRHM off of RHM is the proper way to extend this experiment. Along the way, we have realized that there were following difficulties we need to overcome:

1. In big configurations like number of layers equal to 3 and feature count equal to 12, resultant training dataset size was over 1 TiB if we tried to load it to memory, and more than 100 GB's if we try to save it. We overcame this problem by limiting example counts to 200000 as no model consumed more than that amount of data. Additionally, memory mapping the result to a file enabled us to properly load and execute the code on the dataset.
2. Modifying the code was really difficult because code that we started our implementation had no comments. Changing the RHM to SRHM
3. Training models where number of layers in the SRHM took extremely long time. Even with A100 GPU's, some training runs were taking hours, and each experiment has around 16 runs to complete if number of layers is 3. This can be resolved by increasing batch size, but authors sticked to batch size of 4, and to replicate results we have sticked to that.

Our current results took around 1 day of training on one A100 40GB GPU.

### 4.2 Bugs in the SRHM and Sensitivity Calculations

As we implemented SRHM, we have implemented the following to ensure that we were introducing no bugs to the system:

1. Assert SRHM data is balanced on labels, has proper dimensions, and content of its data.  
2. Assert SRHM with diffeo or synonym exchange applied has different data points compared to original SRHM.
3. Assert training run gradients are not exploding.
4. Assert training accuracy and test accuracy are behaving properly (staying constant, not increasing very rapidly etc.)
5. See that train run graphs look like low accuracy for a long time, then increasing around sample complexity value. You can inspect Experiment 1A graph.
6. Assert outputs coming from CNN/LCN layer hooks are normal.

Even though we have controlled on these problems, we were unable to resolve following bugs:

1. Sensitivity calculations never go below 0.7 or stay close to 1.
2. Sometimes, sensitivity calculations go to inf or nan. We think this is caused by some datapoints being equal to eachother between a SRHM and synonym or diffeomorphism exchange applied version of SRHM.

### 4.3 Adapting to Efficientnet, Resnet etc.

Adapting the SRHM to networks like Efficientnet was a problem we could not solve. We have assumed the following approach, but we were not able to solve bugs between sparsity calculations and dimension conversions:

1. Networks like Resnet expect an input like `(B, N, H, W)` where H and W is usually 224. Form of data from SRHM is like `(B, m^2)` for decimal encoding and `(B, num_classes+1, m^2)` for onehot encoding.
2. One way to view the data is `(B, 1, num_classes+1, m^2)`, but for experiment 4, this causes actual dimensions `(4, 1, 11, 36)` which is unfit for Resnet.
3. Another way to view the data is `(B, 1, m, m)` while using decimal form. Then this can be extrapolated to `(B, 1, 224, 224)` to give inputs to Resnet. However, repeated interpolations at synonym calculations caused data corruption, and this is not the best way to adapt an architecture like Resnet to our problem.
4. Even after a possible implementation, bugs about sensitivity calculations would have prevented us from continuing.

### 4.4 What We Have Learned

Late into the project, we have realized that authors were saving network states along with accuracies. We could have saved network states for a set number of accuracies, then evaluate sensitivities on saved networks. This would have occupied a very large amount of memory, but we would not spend long times on training.

## 5. References

Main Paper: `How Deep Networks Learn Sparse and Hierarchical Data: the Sparse Random Hierarchy Model`

```bibtex
@InProceedings{pmlr-v235-tomasini24a,
  title =   {How Deep Networks Learn Sparse and Hierarchical Data: the Sparse Random Hierarchy Model},
  author =       {Tomasini, Umberto Maria and Wyart, Matthieu},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages =   {48369--48389},
  year =   {2024},
  editor =   {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume =   {235},
  series =   {Proceedings of Machine Learning Research},
  month =   {21--27 Jul},
  publisher =    {PMLR},
  pdf =   {https://raw.githubusercontent.com/mlresearch/v235/main/assets/tomasini24a/tomasini24a.pdf},
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

## Contact

- Burak Erinç Çetin - <erinc.cetin@metu.edu.tr>
- Emin Sak - <sak.emin@metu.edu.tr>
