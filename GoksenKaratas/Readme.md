# Robust Emotion Recognition in Context Debiasing

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

Emotion recognition plays a crucial role in human-computer interaction systems. In the scope of emotion recognition, the contextual features of images can help the model identify the emotion classes effectively. For this matter, context-aware emotion recognition (CAER)[2] approaches have been developed in a way that incorporates such context features by specifically focusing on them. However, CAER methods often suffer from context bias, leading to suboptimal results. The negative effects of such context bias can be so powerful that in some cases, the emotion classes of the given images can be predicted solely depending on the contextual features regardless of the actual object (humans in this case).

<p align="center"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/image1.png" alt="Emotion Recognition Context Bias" width="600" height="600"> </p>

This paper introduces a novel framework, **Counterfactual Emotion Inference (CLEF)** [1], to mitigate context bias and improve robustness in CAER tasks. Specifically, a generalized causal graph is formulated to shed light to the causal relationships between variables in CAER flow. Following the graph, a non-invasive context branch is integrated to capture the adverse direct effect caused by the context bias. 

This repository contains the re-implementation of the paper "Robust Emotion Recognition in Context Debiasing", authored by Dingkang Yang, Kun Yang, Mingcheng Li, Shunli Wang, Shuaibing Wang, Lihua Zhang. The paper was published in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

The primary aim of this repository is to ensure the reproducibility of the results presented in the paper by implementing the CLEF framewrok and conducting experiments as described in the paper.

---

## 1.1. Paper summary

Existing Context-Aware Emotion Recognition (CAER) approach uses both the subject and the surrounding context to predict emotions of a given image. While contextual features can enhance the accuracy of predictions, they might also cause context bias to be included, where spurious correlations lead to inaccurate predictions.

Some drawbacks of CAER:
* Dependence on context results in biased predictions.
* Performance bottleneck (by forcing the models to rely on spurious correlations between background contexts and emotion labels in likelihood estimation)
  
To address these issues, the paper proposes the Counterfactual Emotion Inference Framework (CLEF), which uses causal inference to separate helpful prior and harmful bias. By removing the direct context effect (bias) and preserving only the indirect context effect (helpful prior), CLEF ensures more robust emotion predictions.
In the scope of the paper, the authors extend the pre-existing CAER architecture to capture the indirect causal effects and reduce their negative effects on prediction performance. They construct a causal graph to formulate the relationships among variables in CAER. Using the causal graph, CLEF propose a non-invasive context branch aimed at capturing the direct effects of context bias. During inference, CLEF computes both factual and counterfactual outcomes to remove the direct context effect from the total causal effect. This process ensures that the model's predictions are less influenced by biased contextual information and thus enhances emotion recognition performance.

For evaluating the performance of the proposed framework, experiments are conducted on two large-scale image-based CAER datasets, including EMOTIC[3] and CAER-S[4]. The results for the five baseline models EMOT-Net[3], CAER-Net[4], GNN-CNN[5], CD-Net[6], and EmotiCon[7] show that integrating CLEF increases performance across all approaches.

---

# 2. The method and our interpretation

## 2.1. The original method

The proposed method formulates the problem as a causal graph and uses the implied relations to mitigate the direct effects of context bias while preserving the desired undirect prior.
First, the authors start by defining the preliminary relations obtained from causal graph theory:
<p align="center"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/CausalGraphs.png" alt="Emotion Recognition Context Bias" width="400"> </p>

To architecturally define the framework, the authors introduce an additional context branch to the pre-existing CAER architecture in order to denote the contextual bias. Once the emotion recognition problem is formulated as a causal graph, the given relations became useful for extracting direct effects of branches.
While extending the factual component, they also introduce a counterfactual component with a very similar architecture as the factual one. The main motivation behind this choice is to incorporate the extracted realations from the causal graphs and be able to govern the effects of the contextual features. 

<p align="center"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/Architecture.png" alt="Architecture" width="900"> </p>

The causal graph formulation of vanilla CAER framework is provided as follows:

* Input images X 
* Subject features S
* Context features C 
* Ensemble representations E 
* Emotion predictions Y  

**Link X → C → Y** reflects the shortcut between the original inputs X and the model predictions Y through the harmful bias in the context features C.

**Link C ← X → S** portrays the total context and subject representations extracted from X via the corresponding encoders in vanilla CAER models. 

**Link C/S → E → Y** captures the indirect causal effect of C and S on the model predictions Y through the ensemble representations E.

By examining the CAER causal graph links, the additional relations are introduced into the causal graph of CLEF. To mitigate the interference of the harmful context bias on model predictions, they exclude the biased direct effect along the link X → C → Y. The causality in the factual scenarios is formulated as follows:
<p align="left"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/EQ5.png" alt="EQ5" width="400"> </p>

This reflects the confounded emotion predictions since it suffers from the undesired effects of the pure context bias. To be able to fix  distinct causal effects in the context semantics, the Total Effect (TE) of C = c and S = s are calculated:
<p align="left"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/EQ6.png" alt="EQ6" width="400"> </p>

where c∗ and e∗ represent the non-treatment conditions for observed values of C and E, where c and s leading to e are not given.
Next, the Natural Direct Effect (NDE) for the harmful bias in context semantics is estimated in order to clarify the causalty between them
<p align="left"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/EQ7.png" alt="EQ7" width="400"> </p>

To exclude the explicitly captured context bias in NDE, Total Indirect Effect (TIE) is estimated by subtracting NDE from TE
<p align="left"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/EQ9.png" alt="EQ9" width="400"> </p>
The resulting TIE is than used as the unbiased prediction during inference.


**Implementation Instantiation**


CLEF’s predictions consist of two parts: 
* the prediction `Y_c(X) = N_C(c|x)` of the additional context branch (i.e., \(X → C → Y\))
* `Y_e(X) = N_{C,S}(c, s|x)` of the vanilla CAER model (i.e., \(X → C/S → E → Y\)). 

The context branch is a neural network like ResNet to receive context images with masked subjects. The masking operation contains masking out the subject by assigning the pixel values of the subject to zero. Here, masking forces the network to focus on pure context semantics for estimating the direct effect. 

Once the values for both branches are calculated, a pragmatic fusion strategy φ(·) is introduced to obtain the final score Y_{c,e}(X):
<p align="left"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/Fusion.png" alt="fusion" width="400"> </p>

**Training**

The authors state that as a universal framework, they have selected multi-class classification problem as an example to adopt the cross-entropy loss as the objective function. The general loss is calculated by summing the losses of two branches independently:

<p align="left"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/LossSummation.png" alt="loss" width="400"> </p>


For the counterfactual branch, the imagined Y_e*(X) is implemented as a trainable parameter initialized by the uniform distribution since neural networks cannot handle void inputs. It should be noted that the Y_e*(X) is shared by all samples. To be able to regularize Y_e*(X), since inappropriate values can cause TIE to be dominated by TE or NDE, Kullback-Leibler divergence is employed.
<p align="left"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/KL.png" alt="kl" width="400"> </p>

After integrating all components together, the final loss is formulated as the following:
<p align="left"> <img src="https://github.com/CENG501-Projects/CENG501-Fall2024/blob/main/GoksenKaratas/images/FinalLoss.png" alt="finalloss" width="400"> </p>

---


## 2.2. Our interpretation
The implementation details provided by the authors appear to be sufficient for re-implementing the paper at the time of creating this README file. This section may be updated as the implementation study progresses.

# 3. Experiments and results

## 3.1. Experimental setup

In this paper, experiments are conducted on two large-scale image-based CAER datasets, namely EMOTIC and CAER-S. The evaluation of the proposed framework is performed using five representative methods with completely different structures and contextual modelling paradigms:

* EMOT-Net[3]
* CAER-Net[4]
* GNN-CNN[5]
*  CD-Net[6]
*  EmotiCon[7]

The effectiveness of the five given CLEF-based methods are compared against existing state-of-the-art models such as:
* HLCR [8]
* TEKG [9]
* RRLA [10]
* VRD [11]
* SIB-Net [12]
* MCA [13]
* GRERN [14]



## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

[1] Yang, Dingkang, et al. "Robust Emotion Recognition in Context Debiasing." [arXiv:2403.05963](https://arxiv.org/abs/2403.05963)

[2] Ronak Kosti, Jose M Alvarez, Adria Recasens, and Agata Lapedriza. Emotion recognition in context. In Proceedings of the IEEE/CVF Conference on computer Vision and Pattern Recognition (CVPR), pages 1667–1675, 2017. 1, 2, 3, 4, 6

[3] Ronak Kosti, Jose M Alvarez, Adria Recasens, and Agata Lapedriza. Context based emotion recognition using emotic dataset. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(11):2755–2766, 2019. 1, 2, 3, 5, 6, 7, 8

[4] JiyoungLee,SeungryongKim,SunokKim,JunginPark,and Kwanghoon Sohn. Context-aware emotion recognition net- works. In Proceedings of the IEEE/CVF International Con- ference on Computer Vision (ICCV), pages 10143–10152, 2019. 3,4,5,6,7,8

[5] Minghui Zhang, Yumeng Liang, and Huadong Ma. Context- aware affective graph reasoning for emotion recognition. In IEEE International Conference on Multimedia and Expo (ICME), pages 151–156. IEEE, 2019. 2, 3, 6, 7

[6] Zili Wang, Lingjie Lao, Xiaoya Zhang, Yong Li, Tong Zhang, and Zhen Cui. Context-dependent emotion recog- nition. Journal of Visual Communication and Image Repre- sentation, 89:103679, 2022. 2, 3, 6, 7, 8

[7] Trisha Mittal, Pooja Guhan, Uttaran Bhattacharya, Rohan Chandra, Aniket Bera, and Dinesh Manocha. Emoticon: Context-aware multimodal emotion recognition using frege’s principle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14234–14243, 2020. 1, 2, 3, 4, 6, 7

[8] Willams de Lima Costa, Estefania Talavera, Lucas Silva Figueiredo, and Veronica Teichrieb. High-level context rep- resentation for emotion recognition in images. In Proceed- ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop (CVPRW), pages 326–334, 2023. 2, 3, 6, 7

[9] Jing Chen, Tao Yang, Ziqiang Huang, Kejun Wang, Me- ichen Liu, and Chunyan Lyu. Incorporating structured emo- tion commonsense knowledge and interpersonal relation into context-aware emotion recognition. Applied Intelligence, 53 (4):4201–4217, 2023. 2, 3, 6, 7

[10] Weixin Li, Xuan Dong, and Yunhong Wang. Human emo- tion recognition with relational region-level analysis. IEEE Transactions on Affective Computing, 2021. 2, 3, 6, 7

[11] Manh-Hung Hoang, Soo-Hyung Kim, Hyung-Jeong Yang, and Guee-Sang Lee. Context-aware emotion recognition based on visual relationship detection. IEEE Access, 9: 90465–90474, 2021. 2, 3, 6, 7

[12] Xinpeng Li, Xiaojiang Peng, and Changxing Ding. Se- quential interactive biased network for context-aware emo- tion recognition. In IEEE International Joint Conference on Biometrics (IJCB), pages 1–6, 2021. 2, 3, 6, 7

[13] DingkangYang,ShuaiHuang,ShunliWang,YangLiu,Peng Zhai, Liuzhen Su, Mingcheng Li, and Lihua Zhang. Emotion recognition for multiple context awareness. In Proceedings of the European Conference on Computer Vision (ECCV), pages 144–162, 2022. 2, 3, 6, 7

[14] Qinquan Gao, Hanxin Zeng, Gen Li, and Tong Tong. Graph reasoning-based emotion recognition network. IEEE Access, 9:6488–6497, 2021. 2, 3, 6, 7

# Contact

Pervin Mine Gökşen

mine.goksen@metu.edu.tr

GitHub: https://github.com/MineGoksen


Yahya Bahadır Karataş 

bahadir.karatas@metu.edu.tr

GitHub: https://github.com/bahadirkaratas
