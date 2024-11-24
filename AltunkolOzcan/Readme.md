# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

For our CENG 501 Deep Learning term project, we implement the following paper on neural system identification and control: "Neural Optimal Control using Learned System Dynamics" [1]. The paper was published in Internation Conference on Robotics and Automation 2023, one of the most renowned and prestigious conferences in robotics. 

The paper introduces neural networks to identify unknown nonlinear dynamics. Next, a neural optimal controller is developed using the identified dynamics. The results for both identification and control are demonstrated in four systems: Acrobot, Dubins car, Cartpole and Quadrotor. Since the primary goal of this project is to delve deep into the details of neural network implementations, we proceed to regenerate results incrementally. This means we implement system identification of some of these systems in the beginning. Later, as time permits, we implement the controllers for the systems identified. Once some systems have been studied end-to-end, we move on to investigate the remaining systems step-by-step.

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

![](/../main/AltunkolOzcan/images/overview.png)
*Figure X: Original method overview*

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


# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
