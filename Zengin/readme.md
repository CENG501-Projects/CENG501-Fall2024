# On the Faithfulness of Vision Transformer Explanations



## 1. Introduction

Vision Transformers (ViTs), introduced by Dosovitskiy et al. (2021) in *"An Image is Worth 16x16 Words"*, have revolutionized computer vision tasks by leveraging self-attention mechanisms originally developed for natural language processing. Despite their success, their inherently opaque nature necessitates post-hoc interpretability techniques to provide human-understandable explanations for their predictions.

### Why Interpretability Matters
Post-hoc explanations, such as salience maps, assign scores to input features (e.g., pixels) based on their relevance to a model's decision. However, the "faithfulness" of these explanations—how accurately the salience scores represent the true causal effect of input features—remains underexplored. Faithfulness is critical for trustworthiness, especially in high-stakes applications such as healthcare and autonomous systems.

### Goal of the Paper
The paper *"On the Faithfulness of Vision Transformer Explanations"* addresses this gap by proposing a novel metric, **Salience-guided Faithfulness Coefficient (SaCo)**, to evaluate the faithfulness of explanation methods. The metric quantitatively measures the alignment between salience scores and their actual impact on model predictions.

### Research Questions
1. Are current metrics for evaluating explanation methods sufficient to measure faithfulness in Vision Transformers?
2. How can faithfulness be defined and rigorously evaluated for salience maps?
3. What design improvements enhance the faithfulness of existing explanation methods?

### Structure of This Work
This project aims to reproduce the paper’s results, analyze its methods, and provide insights for improving faithfulness metrics in the context of ViTs. We focus on re-implementing SaCo, comparing it with existing metrics, and experimenting with various explanation methods to validate its claims.

---

### 1.1. Paper Summary

The main contributions of the paper are:
1. **Faithfulness Definition and Metric**: SaCo evaluates salience maps by comparing the influence of pixel subsets with different salience scores on model predictions. Unlike existing metrics such as AOPC or AUC, SaCo assesses individual pixel subsets rather than cumulative perturbations.
2. **Empirical Findings**: Using ViT-B, ViT-L, and DeiT-B models across CIFAR-10, CIFAR-100, and ImageNet datasets, SaCo demonstrates:
   - Current metrics fail to differentiate between meaningful explanations and random attributions.
   - Attention-based methods with gradient and layer aggregation achieve higher faithfulness scores.
3. **Theoretical Contributions**: SaCo leverages statistical comparisons of salience subsets inspired by the Kendall τ statistic, ensuring scale invariance and robustness against perturbation order.

---

### 1.2. Related Work

#### Post-hoc Explanation Methods
1. **Gradient-based Methods**: 
   - *Integrated Gradients* (Sundararajan et al., 2017) and *Grad-CAM* (Selvaraju et al., 2017) use gradients to assign salience scores.
   - **Limitations**: Dependence on local gradients, which may not reflect the model's global reasoning.
2. **Attribution-based Methods**:
   - *Layer-wise Relevance Propagation* (Binder et al., 2016) propagates relevance scores through layers.
3. **Attention-based Methods**:
   - Methods like *Raw Attention*, *Rollout*, and *ATTCAT* utilize attention weights as explanations.

#### Faithfulness Evaluation Metrics
- **AUC**: Measures model performance under cumulative pixel removal.
- **AOPC**: Quantifies output variations after perturbation.
- **Comprehensiveness**: Measures importance of pixels with lower salience.
- **Limitations**: These metrics do not evaluate the magnitude differences between salience subsets, failing to validate faithfulness.

---

## 2. The Method and Our Interpretation

### 2.1. The Original Method

#### Salience-guided Faithfulness Coefficient (SaCo)

The core assumption is that salience scores should correlate with the impact of input features on predictions. Mathematically:

1. **Salience Map**:  
   Given an input image \(x\), model prediction \(y = \hat{f}(x)\), and explanation method \(E\), the salience map \(M(x)\) assigns a score to each pixel:
   $$
   M(x) = \{m_{ij} \mid m_{ij} \in \mathbb{R}, \, \forall (i, j) \in x\}
   $$
   where \(m_{ij}\) is the salience score of pixel \((i, j)\).

2. **Subset Partitioning**:  
   Pixels are divided into \(K\) subsets \(G_k\) based on salience ranking:
   $$
   G_k = \{(i, j) \mid (k-1)\frac{HW}{K} \leq \text{rank}(m_{ij}) < k\frac{HW}{K}\}
   $$
   where \(HW\) is the total number of pixels.

3. **Perturbation Impact**:  
   Replace pixels in \(G_k\) with their mean value and measure the change in model confidence:
   $$
   \Delta_k = p(y | x) - p(y | R_p(x, G_k))
   $$
   where \(R_p(x, G_k)\) denotes the perturbed image.

4. **Faithfulness Coefficient**:  
   Compare all pairs of subsets \(G_i, G_j\) and accumulate the differences in salience scores:
   $$
   F = \frac{\sum_{i, j} w_{ij} \cdot \text{sign}(\Delta_i - \Delta_j)}{\sum_{i, j} |w_{ij}|}, \quad w_{ij} = s(G_i) - s(G_j)
   $$
   \(F \in [-1, 1]\), where positive values indicate faithful alignment.

---

### 2.2. Our Interpretation

#### Clarifications and Extensions
1. **Perturbation Strategy**:  
   The choice of replacing pixels with mean values is based on prior work but may introduce bias. We explored alternatives, such as Gaussian noise or black-out perturbations, to assess their impact on SaCo scores.
2. **Subset Size \(K\)**:  
   The granularity of pixel subsets influences faithfulness evaluation. Smaller \(K\) captures more subtle salience differences but increases computation.
3. **Cross-layer Aggregation**:  
   SaCo's success with attention-based methods highlights the importance of integrating gradient information and multi-layer attention. We hypothesize that these designs capture more global model behavior.

#### Connections to Related Metrics
- **Kendall \( \tau \) Statistic**:  
  SaCo’s pairwise comparisons are inspired by the Kendall \( \tau \) statistic for rank correlation, ensuring robustness in salience evaluations.
- **Scale Invariance**:  
  Unlike AOPC or AUC, SaCo is unaffected by normalization or scaling of salience scores, ensuring consistent evaluations across methods.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
