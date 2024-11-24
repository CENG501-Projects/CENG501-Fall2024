# Neural Optimal Control using Learned System Dynamics

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects. The commit history of the project is available through [this github repo link](https://github.com/emir-altunkol/CENG501-Project).

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

For our CENG501 Deep Learning term project, we implement the following paper on neural system identification and control: "Neural Optimal Control using Learned System Dynamics" [1]. The paper was published in International Conference on Robotics and Automation 2023, one of the most renowned and prestigious conferences in robotics. 

The paper introduces neural networks to identify unknown nonlinear dynamics. Next, a neural optimal controller is developed using the identified dynamics. The results for both identification and control are demonstrated in four systems: Acrobot, Dubins car, Cartpole and Quadrotor. Since the primary goal of this project is to delve deep into the details of neural network implementations, we proceed to regenerate results incrementally. This means we implement system identification of some of these systems in the beginning. Later, as time permits, we implement the controllers for the systems identified. Once some systems have been studied end-to-end, we move on to investigate the remaining systems step-by-step.

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

Optimal control is fundamental for robotics tasks such as navigation and locomotion, enabling systems to achieve desired outcomes while minimizing the defined costs. Common approaches to solving these problems include trajectory optimization methods, such as Nonlinear Model Predictive Control (NMPC), and numerical solutions to Hamilton-Jacobi-Bellman (HJB) equations. However, these methods face significant limitations: NMPC computes optimal control for a single initial state, requiring lengthy recomputations for each new state, which is impractical for real-time applications. On the other hand, solving HJB equations numerically necessitates grid-based discretization of the state space, limiting scalability to high-dimensional problems. This paper proposes a grid-free method for addressing these challenges. Unlike traditional local methods, which are limited to optimizing control laws for individual initial states, the proposed approach generates control laws for a wide range of initial states sampled from a large portion of the state space. This capability bridges the gap between local and global methods, and provides a compromise between these two approaches.

Figure 1 provides an overview of the proposed method, which offers a two-step neural network-based solution to optimal control problems. In the first step, system identification is performed by training a neural network—a simple multi-layer perceptron (MLP)—using Neural Ordinary Differential Equations (Neural ODEs) on uniformly sampled state-space data. In this step, the network learns the system's dynamics from the data. In the second step, the learned dynamics are used to train two additional neural networks simultaneously: one representing the value function and the other representing the controller. These networks are trained synchronously by minimizing multiple loss functions, as illustrated in Figure 1. The details about the calculation of the loss functions and some parameters of the training processes are given in the paper.

<p align="center">
  <img src="/../main/AltunkolOzcan/images/overview.png" alt="Overview of the proposed optimal control method">
</p>

<p align="center">Figure 1: Overview of the proposed optimal control method [1]</p>

Hence, even if the true system dynamics is not known, the proposed method is able to learn the dynamics and train the controller accordingly. As many papers in the literature suggest model-based reinforcement learning (MBRL) strategies for the systems with unknown true dynamics, the performance of this method is compared with multiple MBRL methods proposed in the literature. During the experiments, the original state transition function f is utilized to assess the performance of various controllers. The results show that the proposed neural controller outperforms a comparable neural controller from the literature and several MBRL-based methods across multiple control tasks, including Dubins Car, Cartpole, Acrobot, and Quadrotor systems.

# 2. The method and our interpretation

## 2.1. The original method

@TODO: Explain the original method.

### 2.1.1 System Identification

Suppose the system obeys the following time independent dynamical equation:

```math
\dot{x} = f(x, u),
```
where $x$ is the state of the system and $u$ is the input. The first goal is to train a neural network to demonstrate the relationship between $x$ and $\dot{x}$; namely, to learn the $f$ function. 

The original method uses a three layer MLP $f_{\theta}$ to represent the $f$ function. The network is trained in a supervised fashion using using data samples of the form $ (x, u, f(x,u))$. The activation functions are sine functions, although the paper also provides results with other activation functions such as tanh and ReLU for comparison. 

In addition to the values of $f$, $\nabla{f}$ are used to supervise training. The authors have found this can help generate smoother learned systems dynamics. Consequently, the loss function for system identification is set as 

```math
\mathfrak{L}_{sys-id} = \sum_{i} || f_{\theta}(x_i,u_i) - f(x_i,u_i)|| + || \nabla{f}_{\theta}(x_i, u_i) - \nabla{f}(x_i, u_i) ||
```

### 2.1.1 The Value Function and Controller Design

According to the paper, so called control Hamiltonian $H$ is used for optimal control. $H$ is defined as

```math
H(x_t, \lambda_t, t) = \min_{u_t} \left\{ L(x_t, u_t, t) + \lambda_t^\top \cdot f(x_t, u_t, t) \right\},
```
where $L(x_t, u_t, t)$ is the running cost for control and $\lambda_t = \nabla_x V(x_t, t)$. Here, $V$ is known as the value function and it represent the optimal cost-to-go. Cost-to-go is initialized as $G(x_f)$, the terminal cost. The method in the paper uses a neural network to learn $V$ as well as the optimal controller.

In theory, what is known as Pontryagin’s Maximum Principle (PMP) is used to find the optimal control signal. The paper provides a neural network approach for cases where the analytical solution is hard, especially with unknown dynamics. The authors name their approach as Control with Neural Dynamics (CND).  

The controller network is made up of an MLP whose last layer is followed by a layer of tanh function in order to scale the input signal to the feasible range. Moreover, the network for the value function is also an MLP with tanh activations and skip connections. The remaining details of the networks, except for the loss functions, are not disclosed in the paper. 

The training stats by sampling random initial states and randomly initialized controller parameters. Trajectories of the system are calculated using the learned dynamics $f_{\theta}$. The value function and the controller network are jointly trained by öinimizing the following cost function:

```math
\mathcal{L} = \alpha_{\text{cost}} \mathcal{L}_{\text{cost}} + \alpha_{\text{HJB}} \mathcal{L}_{\text{HJB}} + \alpha_{\text{final}} \mathcal{L}_{\text{final}} + \alpha_{\text{hamil}} \mathcal{L}_{\text{hamil}}
```

where each term is another cost function defined as

```math
\mathcal{L}_{\text{cost}} = \mathbb{E}_{x_0 \sim \rho} \int_{t_0}^{t_f} L(x_t, u_t, t) \, dt + G(x_f),
```
```math
\mathcal{L}_{\text{HJB}} = \left\lVert \frac{\partial V_{\Phi}(x_t, t)}{\partial t} + H \right\rVert_1,
```
```math
\mathcal{L}_{\text{final}} = \left\lVert V_{\Phi}(x_f, t_f) - G(x_f) \right\rVert_1.
```

The paper declares the parameters used as $\alpha_{cost} = 1$, $\alpha_{HJB} = 1$, $\alpha_{final} = 0.01$. Adam optimizer is used with a decaying learning rate starting from 0.01.

## 2.2. Our interpretation

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

Although the paper is clear about the many details, there still remain some parts undisclosed and left to the reader to experiment with.

### 2.2.1 Neural ODE MLP

The number of neurons in the layers of the MLP for the system identification task is unclear. We try to adapt the best performing design using several experiments.

The initialization of weights is also left ambiguous. They are randomly initialized by our choice because the weights for the controller network are said to be randomly initialized, as explained in the next section.

### 2.2.2 The Value Function and The Controller Networks

The depth and width of the MLPs used for the value function and the neural controller are not given in the paper. Our strategy is to determine the best architecture by trial and error.

The weights of the controller MLP are initialized randomly, but no information is explicitly given for the network of the value function. We assume no change of style probbaly happened during method development. As a result, we initialize the weights of the MLP randomly also.

### Data Generation

The data sets consist of the triples $ (x, u, f(x,u))$. 

The state and input are said to be sampled uniformly in their respective spaces. However, experimental results actually provide no bounds for the spaces. We prefer to set artifical limits such as $ -5 < u < 5$ to simplify the sampling process.

In addition, it is explained that $f(x,u)$ are generated using the ground truth dynamics. The paper references [2], [3] at this point for Acrobot and Cartpole. However, we currently generate the datasets in MATLAB for the following reasons:

- In MATLAB, we can directly set bounds to the variables and take samples.
- The simulators in the references are not guaranteed to uniformly sample the state space.
- Since the neural networks do not accept time as input, no actual simulation is needed. It is sufficient to pick a point and calculate the derivatives at that point until enough data is present.

The system identification network is supervised by the gradients of the function $f$. We use the term gradient cautiously here because $x$ is a vector for any multidimensional system and this means $f$ is in fact multi-output. Therefore, we believe Jacobian would be a better use of terminology. In that case, the relevant cost function must use a matrix norm, not a vector norm. In out implementation, Frobenius norm is used as the matrix norm.

The direct calculation of the Jacobian is difficult. So, unlike the data generation for $f(x,u)$, we do not use MATLAB and the ground truth equations. Instead, Pytorch's automatic differentiation engine [4] will likely be used to generate the ground truth Jacobian of $f$. 

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

[1] S. Engin and V. Isler, "Neural Optimal Control using Learned System Dynamics," 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, 2023, pp. 953-960, doi: 10.1109/ICRA48891.2023.10160339.

[2] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

[3] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018.

[4] "Autograd: Automatic Differentiation," PyTorch Tutorials, Accessed: Nov. 24, 2024. [Online]. Available: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html


# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
