# **Improving Generalization via Meta-Learning on Hard Samples**

## 
# 1. Paper Summary
## **1.1. Introduction**

Learned Reweighting is an approach that utilizes assigned weights on dataset to optimize validation dataset representation. The authors in this paper show that hard-to-classify instances has both theoretical connection and strong empirical evidence to generalization. 

Their proposed methods outperform a wide range of baselines on various train and test datasets.

The distribution of train and validation datasets were used in terms of their losses. Recent studies showed the importance of weighted training instances and their effects on overall training performance.

Using "clean" validation data with bilevel optimization can overcome significant amounts of label noise in training data.

The main claim of this paper is that if they create a dataset with hard to classify instances, the classifier generalization improves. They named their method Meta-Optimized Learned Reweighting (MOLERE) where the partition and LRW classifier are jointly classified. They propose the following contributions:

1) Proposing the statement of validation set optimization in learned reweighting  (LRW) classifiers for improved generalization.   
2) Simplifying nested optimization into a min-max game between two auxiliary networks: a "splitter" that finds the hardest samples, and a "reweighter" that minimizes loss on those samples using LRW.  
3) Showing strict accuracy ordering of LRW Models based on validation set: easy \< random \< hard, which demonstrates the importance of optimizing LRW validation sets. Gaining clearly on ERM accross various domain datasets, outperforming various baselines.  
4) Extending results to harder samples as a significant implier on higher generalization.

## **1.2. Related work**

* **Importance reweighting for Robustness:**

Most of the works in this field focused on avoiding noisy labels on training set by comparing the loss of given instance on a clean validation set. This procedure might include learning a per-instance free parameter or learning a simple MLP network to estimate an importance to the instance based on the obtained loss value. Finding a clean validation set in this scenario appears to be a challenge in practice. A Meta-Learning based re-weighting scheme, namely Fast Sample Reweighting, is proposed against this impracticality to generate a pseudo-clean validation set. Same paper also proposed approximations that decrease the computational complexity of the training procedure. A new study named RHO-loss proposes that if only worthy points for training are selected, generalization property of the model would increase. The 'worth' in this context is calculated as the difference between training loss and a hold-out set loss.

Recent Line of work focuses on arranging the weighting based on the context of an instance in relation to overall dataset distribution. There are some studies suggesting target domain data in the validation set can improve domain shift performance of classifiers. 

* **Meta-Learning:**

Sample reweighting task is conceptually similar to Meta-learning, particularly Model-Agnostic Meta-Learning (MAML) approach, which learns a single set of parameters that can be tailored to multiple tasks through optimization on the target domain. Learning the parameters involve a bilevel optimization problem that is similar to the problem proposed in this setting.

* **Probabilistic Margins:**

Recent work showed that probabilistic margins are utilizable in multi-class problems to improve the neural network performance on adversarial examples. Reweighting training instances inversely related to the probability margin of each instance, in presence of an adversarial attack, is proposed.

* **Just Train Twice:**

A recent work proposed to make a 2-level training on ERM models to improve the sensitivity of ERM models towards certain groups. After training the ERM normally in the first stage, the method amplifies the losses of incorrectly classified terms in the aggregate loss function. This can be viewed as amplifying the importance of low margin examples.

## 

## **1.3. Preliminaries: learned reweighting**

This section of the paper outlines a framework for classifier learned reweighting (LRW). The basic idea is to reweight training data according to instance-specific weights in order to get better performance from a classifier on a validation dataset. Optimization of a two-level (or bi-level) objective function is used to accomplish the reweighting.

Two datasets are used in the basic LRW formulation. LRW learns a classifier with parameters θ¸ and an instance-wise weighting function that minimizes the following bi-level objective given a desired loss function.

Notations:

* ![](https://github.com/Sinasi3/Sinasi3/blob/main/3not1.jpg?raw=true): N instance training dataset  
* ![](https://github.com/Sinasi3/Sinasi3/blob/main/3not2.jpg.png?raw=true): M instance training dataset  
* ![](https://github.com/Sinasi3/Sinasi3/blob/main/3not3.jpg.png?raw=true): Loss function  
* ![](https://github.com/Sinasi3/Sinasi3/blob/main/3not4.jpg.png?raw=true): Classifier  
* ![](https://github.com/Sinasi3/Sinasi3/blob/main/3not5.jpg.jpg?raw=true): Instance-wise weighting function

### **1.3.1 Bi-Level optimization objective**

![A group of mathematical symbolsDescription automatically generated](https://github.com/Sinasi3/Sinasi3/blob/main/3not6.jpg.jpg?raw=true)

* **Training Objective** seeks to determine which classifier parameters, ![](https://github.com/Sinasi3/Sinasi3/blob/main/3.1not1.jpg?raw=true), minimize the weighted training loss, with weights ![](https://github.com/Sinasi3/Sinasi3/blob/main/3.1not2.jpg?raw=true)and adjusting the significance of each training instance. The classifier will be guided toward settings that will yield good results on the validation data using this weighting function.  
* **Validation objective** is to optimize ϕ , the instance-specific weights, by minimizing the unweighted validation loss. This step indirectly influences the inner optimization(training objective) by reweighting the training data to influence the learnt classifier, which strengthens the model's generalization on validation data.


  

### **1.3.2.  Intuition and insights**

* The algorithm can indirectly modify training samples to enhance generalization because the training loss is weighted and the validation loss is unweighted.  
* The model effectively treats validation performance as a "target for generalization" by iteratively updating ϕ  and 𝜃 to match the weighted training distribution with the validation distribution.

## **2. Method and Our Interpretation**
## **2.1. Proposed Metohd: MOLERE(Meta-Optimization of the Learned Reweighting)**

The hypothesis presented here is that **generalization in supervised learning** can be improved by combining two key ideas:

1. **Learned-Reweighting (LRW) Classifier:** As was previously mentioned, an LRW classifier optimizes validation performance by modifying the significance of training data.  
2. **Optimized Validation Set:** By carefully choosing a validation set that emphasizes certain properties, especially "hard" samples (challenging instances for the model to classify), the reweighting process is expected to lead to better generalization.  
     
   

First, we choose a validation set of hard examples, and then we use this validation set for LRW to train a classifier on the remaining data. When the aforementioned concept is formalized, it results in a combined optimization issue of data partitioning (train, validation), and LRW training. This is because the difficulty of instances is defined by the learnt model itself. Below, we outline the formal issue.  
![A group of mathematical symbols](https://github.com/Sinasi3/Sinasi3/blob/main/4not.jpg?raw=true)

                                                                   
Notations:

* ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.1.jpg?raw=true) : Validation set  
* ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.2.jpg?raw=true): Training set  
1. **Outer Level:** Optimize the splitting function ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.3.jpg?raw=true) , determining the training set ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.4.jpg?raw=true)and validation set ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.5.jpg?raw=true) split. In reality, we place an extra restriction on the size of the validation set: ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.6.jpg?raw=true), where δ is a predetermined fractional constant.  
2. **Middle Level:** Given the validation set, find the optimal instance-wise weights ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.7.jpg?ra6w=true) to minimize the weighted training loss.  
3. **Inner Level:** Finally, minimize the weighted loss with respect to model parameters θ¸.

### **2.1.1. Objective and generalization**

In order to obtain theoretical understanding of generalization, the study further investigates the asymptotic behavior of MOLERE as the dataset size increases (𝑁 + 𝑀 → ∞). A weighting function ϕ(·)   that depends on both x and y was assumed. 

Suppose:

* ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.1in1i.jpg?raw=true) 
* ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.1in2si.jpg?raw=true)   
* Domains of ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.1in3ü.jpg?raw=true) are very large

  Then the tri-level optimization problem we discussed at 1.2. can be reduced to:

![A close up of symbolsDescription automatically generated](https://github.com/Sinasi3/Sinasi3/blob/main/4.1in4ü.jpg?raw=true) 

**Proof:**

* **STEP1**  
* Molere MOLERE involves solving two nested optimization problems are equivalent:

![A close-up of a math equationDescription automatically generated](https://github.com/Sinasi3/Sinasi3/blob/main/4.1in5i.jpg?raw=true) 

* The weighting function ϕ(x,y), which modifies the probability distribution Q over the dataset, determines ![](https://github.com/Sinasi3/Sinasi3/blob/main/1.png?raw=true).  
* **STEP2**  
* The problem becomes simpler if  ![](https://github.com/Sinasi3/Sinasi3/blob/main/2.png?raw=true). When ![](https://github.com/Sinasi3/Sinasi3/blob/main/3.png?raw=true) is selected, the constraint is as follows: ![](https://github.com/Sinasi3/Sinasi3/blob/main/4.png?raw=true)

* Set   𝜙![](https://github.com/Sinasi3/Sinasi3/blob/main/5.png?raw=true) and select ![](https://github.com/Sinasi3/Sinasi3/blob/main/6.png?raw=true) to minimize over measurable functions if  ![](https://github.com/Sinasi3/Sinasi3/blob/main/7.png?raw=true).  
* **STEP3**  
* Using a weighting function ![](https://github.com/Sinasi3/Sinasi3/blob/main/8.png?raw=true) that matches probabilities on intersections and sets ![](https://github.com/Sinasi3/Sinasi3/blob/main/9.png?raw=true)elsewhere, the same logic applies to any ![](https://github.com/Sinasi3/Sinasi3/blob/main/10.png?raw=true).		  
* **STEP4**  
* Finding the hardest subset S of size 6(N+M) is the definition of the tri-level optimization, which reduces to:

  ![A close up of symbolsDescription automatically generated](https://github.com/Sinasi3/Sinasi3/blob/main/11.png?raw=true)

## **2.1.2.Efficient algorithm for validation optimization**

Creating a tractable meta-optimization method has two technical difficulties: 

* How can we figure out which instances belong to the validation and training sets?   
* How may the tri-level objective be solved effectively?

### **2.1.3. Soft data assignment**

* This article predicts the likelihood that each instance will be a member of the validation set using a "splitter" network.   
* The splitter reduces the cross-entropy between the accuracy of the classifier and its predictions. To guarantee appropriate label distribution and consistency across train-test splits, two regularizers (1) are implemented.

### **2.1.4. Meta-optimization with min-max objective**

The tri-level optimization is reduced by the method to a bi-level formulation, where:

* The splitter dynamically reassigns instances in an attempt to optimize validation set performance.  
* For specified splits, the meta-network reduces validation error.

## **2.1.5. Deriving the update equations**

### **2.1.5.1. Splitter network(Θ)**

* The goal is to maximize loss on hard instances while minimizing loss on the validation set to enhance generalization by accurately predicting the label pair (𝑥,𝑦).

  ![A close-up of a math problemDescription automatically generated](https://github.com/Sinasi3/Sinasi3/blob/main/12.png?raw=true)

### **2.1.5.2. Meta-network (ϕ)**

* The main objective is minimizing the validation error using the classifier's predictions.

  ![A black and white math symbolDescription automatically generated](https://github.com/Sinasi3/Sinasi3/blob/main/13.png?raw=true)

### **2.1.5.3. Classifier network (θ)**

* Optimizing weighted training loss on the training set D′.  
* After e epochs of bi-level setup, ![](https://github.com/Sinasi3/Sinasi3/blob/main/14.png?raw=true) epochs occur on training data, where ![](https://github.com/Sinasi3/Sinasi3/blob/main/15.png?raw=true) accounts for K-times inner loop iterations.

  ![A black and white math symbolsDescription automatically generated with medium confidence](https://github.com/Sinasi3/Sinasi3/blob/main/16.png?raw=true)

## **2.2. Our Interpretation:**
In the paper, some of the approaches were not clearly mentioned. Here is what we saw as unclear, and our interpretations.

- Instructions on how ERM Classifier is constructed was left vague. We have checked the supplementary material, and we have seen different networks are used for different datasets, and we decided that aforementioned ERM Classifier is actually a network trained on a Cross Entropy loss objective to ensure empirical risk minimization.
- Gain over ERM in percentage was frequently used in their graphics, but the authors did not specify how that measure is actually calculated. We ran the experiments at our CIFAR-100 implementation with WRN28-10 network, and we have compared the accuracies. We have seen that gains over ERM is calculated by dividing the current model's accuracy result by the ERM model's accuracy result, and multiply by -1 if the nominator is less than the denominator.
  
Other than these parts, paper was quite detailed and clear about its methods and approaches, proving every mathematical aspect they used and explaining every method they implemented.

## **3\. Experiments and Results**
## **3.1. Experimental Setup**
## **3.2. Running the Code**
## **3.3. Result**
We have conducted the LRW-Hard, LRW-Easy and LRW-Random calculations and compared them over ERM baseline. We have conducted the experiment on CIFAR-100 dataset with Wide-ResNet28-10 implementation that we have provided. Here is our resulting graph:

![WhatsApp Görsel 2024-12-06 saat 17 26 11_4c083e49](https://github.com/user-attachments/assets/bb3f2099-3b83-4530-8d61-60646d8faf95)



## **9.References**

\[1\] Yujia Bao and Regina Barzilay. Learning to split for automatic bias detection. arXiv preprint arXiv:2204.13749, 2022\. 4, 11, 12

## 

