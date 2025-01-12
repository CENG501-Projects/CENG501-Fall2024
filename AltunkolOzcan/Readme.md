# Neural Optimal Control using Learned System Dynamics

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects. The commit history of the project is available through [this github repo link](https://github.com/emir-altunkol/CENG501-Project) (The link will be available after the implementation.).

# 1. Introduction

Nonlinear optimal control problems are encountered in many areas of robotics, and several methods have been developed to solve these problems, including numerical methods, Nonlinear Model Predictive Control (NMPC), and Model-Based Reinforcement Learning (MBRL). However, these approaches have limitations in terms of providing either local or global solutions, requiring space discretization, lacking sample efficiency, and depending on knowledge of the true system dynamics. This paper, Neural Optimal Control using Learned System Dynamics [1], focuses on the control of the systems with unknown dynamics, using a neural controller. The proposed method includes two training steps. First, a multi-layer perceptron (MLP) is trained using Neural ODEs to learn the system dynamics. Then, this learned system dynamics is utilized to train the neural network based controller functions. The controller performance is evaluated using experiments done across four different control systems. The results are compared with the performance of other control methods in the literature, such as multiple MBRL algorithms and another neural network based controller that are developed to solve optimal control problems for the systems with unknown dynamics. According to the results, the proposed method outperforms the other methods in the presented test scenarios. The paper was published in International Conference on Robotics and Automation 2023, one of the most renowned and prestigious conferences in robotics.

The aim of this project is to implement the methods given in the aforementioned paper, and reproduce its results as much as possible. The results for both system identification and control are demonstrated for four systems in the original paper: Acrobot, Dubins car, Cartpole and Quadrotor. Since the primary goal of this project is to delve deep into the details of neural network implementations, we will proceed to regenerate the results incrementally. This means we will first implement system identification for some of these systems in the beginning. Later, as time permits, we will implement the controllers for the systems identified. Once a system have been studied end-to-end, we will move on to investigate the remaining systems, progressing step-by-step.

## 1.1. Paper summary

Optimal control is fundamental for robotics tasks such as navigation and locomotion, enabling systems to achieve desired outcomes while minimizing the defined costs. Common approaches to solving these problems include trajectory optimization methods, such as Nonlinear Model Predictive Control (NMPC), and numerical solutions to Hamilton-Jacobi-Bellman (HJB) equations. However, these methods face significant limitations: NMPC computes optimal control for a single initial state, requiring lengthy recomputations for each new state, which is impractical for real-time applications. On the other hand, solving HJB equations numerically necessitates grid-based discretization of the state space, limiting scalability to high-dimensional problems. This paper proposes a grid-free method for addressing these challenges. Unlike traditional local methods, which are limited to optimizing control laws for individual initial states, the proposed approach generates control laws for a wide range of initial states sampled from a large portion of the state space. This capability bridges the gap between local and global methods, and provides a compromise between these two approaches.

Figure 1 provides an overview of the proposed method, which offers a two-step neural network-based solution to optimal control problems. In the first step, system identification is performed by training a neural network—a simple multi-layer perceptron (MLP)—using Neural Ordinary Differential Equations (Neural ODEs) on uniformly sampled state-space data. In this step, the network learns the system's dynamics from the data. In the second step, the learned dynamics are used to train two additional neural networks simultaneously: one representing the value function and the other representing the controller. These networks are trained synchronously by minimizing multiple loss functions, as illustrated in Figure 1. The details about the calculation of the loss functions and some parameters of the training processes are given in the paper.

<p align="center">
  <img src="/../main/AltunkolOzcan/images/overview.png" alt="Overview of the proposed optimal control method">
</p>

<p align="center">Figure 1: Overview of the proposed optimal control method [1]</p>

The proposed method demonstrates the ability to learn system dynamics and train the controller effectively, even in the absence of true system dynamics. To evaluate its performance, the authors compare their approach against multiple Model-Based Reinforcement Learning (MBRL) methods commonly used for systems with unknown dynamics. During the experiments, the original state transition function $f$ is utilized to assess the performance of various controllers. According to the results presented in the paper, the proposed neural controller outperforms a comparable neural controller from the literature and several MBRL-based methods across multiple control tasks, including Dubins Car, Cartpole, Acrobot, and Quadrotor systems.

# 2. The method and our interpretation

## 2.1. The original method

The method proposed in the paper can be divided into two steps, each of which are explained below.

### 2.1.1 System Identification

Suppose the system obeys the following time independent dynamical equation:

```math
\dot{x} = f(x, u),
```
where $x$ is the state of the system and $u$ is the input. The first goal is to train a neural network to demonstrate the relationship between $x$ and $\dot{x}$; namely, to learn the $f$ function. 

The original method uses a three layer MLP $f_{\theta}$ to represent the $f$ function. The network is trained in a supervised fashion using using data samples of the form $(x, u, f(x,u))$. The activation functions are sine functions, although the paper also provides results with other activation functions such as tanh and ReLU for comparison. 

In addition to the values of $f$, $\nabla{f}$ are used to supervise training. The authors have found this can help generate smoother learned systems dynamics. Consequently, the loss function for system identification is set as 

```math
\mathfrak{L}_{sys-id} = \sum_{i} || f_{\theta}(x_i,u_i) - f(x_i,u_i)|| + || \nabla{f}_{\theta}(x_i, u_i) - \nabla{f}(x_i, u_i) ||
```

### 2.1.2 The Value Function and Controller Design

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

The running and terminal costs are defined as $L(x_t, u_t, t) = (u_t - u^{\*})^TR(u_t - u^{\*})$ and $G(x_f) = (x_f - x^{\*})^T P (x_f - x^{\*})$ respectively. Here, P and R are matrices representing the relative importance of the cost terms, $x^{\*}$ and $u^{\*}$ are the desired state and actions. The authors state that they used $P = I_d$ and $R = 0.01*I_m$, where $I_m$ and $I_d$ are identity matrices. The paper also declares the parameters used as $\alpha_{cost} = 1$, $\alpha_{HJB} = 1$, $\alpha_{final} = 0.01$. Adam optimizer is used with a decaying learning rate starting from 0.01.

## 2.2. Our interpretation

Although the paper is clear about the many details, there still remain some parts undisclosed and left to the reader to experiment with.

### 2.2.1 Neural ODE MLP

The number of neurons in the layers of the MLP for the system identification task is unclear. We try to adapt the best performing design using several experiments.

The initialization of weights is also left ambiguous. They are randomly initialized by our choice because the weights for the controller network are said to be randomly initialized, as explained in the next section.

### 2.2.2 The Value Function and The Controller Networks

The depth and width of the MLPs used for the value function and the neural controller are not given in the paper. Our strategy is to determine the best architecture by trial and error.

The weights of the controller MLP are initialized randomly, but no information is explicitly given for the network of the value function. We assume no change of style probbaly happened during method development. As a result, we initialize the weights of the MLP randomly also.

The paper cites a previous work related to the application of deep learning methods in optimal control problems. We have studied this paper [6] thoroughly and tried to change, adapt to and add to their existing codebase for our specific needs and novelties. 

#### 2.2.2.1 Control of Acrobot

Acrobot is an underactuated double pendulum. The control task is to bring both links to an upright position and keep it there with zero control effort. This corresponds to the target state $[\pi, \pi, 0, 0]$ with target input of $0$. We try to restrict inputs between $250 Nm$ and $-250Nm$. We derive the Hamiltonian and other costs/metrics by assuming the exact $f$ function is known. The control time horizon is 1 second, which consists of 50 control steps. 

The controller and the value function network architectures are unclear in the original paper. The controller network in our implementation is made up of 6 layer MLP with sine and LeakyReLU activation functions in the hidden layers. The last layer of this network is followed by a tanh function as advised in the paper. We experiment with width and depth of this network. The value function network is made up of a ResNet and some quadratic terms added as in [6]. 

We also found use in changing $\alpha$ coefficients of cost terms during training. We first focus on the terminal cost, then change to the coefficients in the paper [1] to focus on optimal control. 

To calculate the various costs, we have come up with methods. The running cost and the terminal cost are set as in [1]. These terms are integrated during training as done in [6]. As a result, we do not expect our network to ever reach zero cost because it is impossible. To calculate the Hamilton Jacobi Bellman equation cost, we evaluate the gradient of the value function network with respect to time by concatanating each state with the corresponding time. The value function network result is used in the calculation of the final cost-to-go cost together with the terminal cost calculation. Differently from the paper [1], we also use the state derivatives of the value function network to match the state derivatives of the cost to go network as in [6]. 

### 2.2.3 Data Generation

The data sets consist of the triples $(x, u, f(x,u))$. 

The state and input are said to be sampled uniformly in their respective spaces. However, experimental results actually provide no bounds for the spaces. We prefer to set artifical limits such as $-5 < u < 5$ to simplify the sampling process.

In addition, it is explained that $f(x,u)$ are generated using the ground truth dynamics. The paper references [2], [3] at this point for Acrobot and Cartpole. However, we currently generate the datasets in MATLAB for the following reasons:

- In MATLAB, we can directly set bounds to the variables and take samples.
- The simulators in the references are not guaranteed to uniformly sample the state space.
- Since the neural networks do not accept time as input, no actual simulation is needed. It is sufficient to pick a point and calculate the derivatives at that point until enough data is present.

The system identification network is supervised by the gradients of the function $f$. We use the term gradient cautiously here because $x$ is a vector for any multidimensional system and this means $f$ is in fact multi-output. Therefore, we believe Jacobian would be a better use of terminology. In that case, the relevant cost function must use a matrix norm, not a vector norm. In out implementation, Frobenius norm is used as the matrix norm.

The direct calculation of the Jacobian is difficult. So, unlike the data generation for $f(x,u)$, we do not use MATLAB and the ground truth equations. Instead, Pytorch's automatic differentiation engine [4] will likely be used to generate the ground truth Jacobian of $f$. 

### 2.2.4 Example Systems

#### 2.2.4.1. Acrobot

One of the systems the paper tests is Acrobot, which is an underactuated double pendulum. Its dynamic equations are given as

```math
\mathbf{M(q)\ddot{q} + C(q,\dot{q})\dot{q} = \tau_g(q) + Bu},
```
where 
```math
\mathbf{q = [\theta_1, \theta_2]^T}, \mathbf{u = \tau}.
```
The resulting state space is four dimensional with single input. 

The paper explains that states and inputs are sampled uniformly from their respective spaces, but it does not disclose the limits of those spaces. Thus, we resort to some made up limits. Here $|u|<1$, $|x_i|<10$. Note that only the initial samples are drawn with respect to these limits and the trajectories may violate them.

The system identification network uses the $f_\theta$ values to train the network. We generate these values with a Python code, which is the converted version of the existing MATLAB code at [5]. First random inputs are generated. Then the inputs are used to evaluate state derivatives together with randomly sampled states. This gets rid of the need to generate an actual trajectory for training data. The random samples with the derivative values at the samples constitute the training dataset. 

For testing, we actually generate a random trajectory this time. This is done by producing random inputs, and using odeint() method of Pytorch to solve the original differential equation with the inputs. Then, at every 20 iterations the neural network is similarly used in the odeint() to generate a predicted trajectory starting from the same initial condition. If the network succesfully manages to capture the time derivatives and if the numerical differential equation solver is accurately tuned for the system, then the results must match. The ODE solver method is selected as the Fourth Order Runge Kutta with 3/8 rule. This solver has a fixed step size. We found it useful because it significantly reduces the testing times. However, the drawback is that the data frequency has to be very high for the method to produce reliable results. Currently, a trajectory of 128 seconds is generated with 8192 data points, corresponding to 64 Hz.

The Neural ODE network consists of three layer MLP with sine activation functions. The input layer has 5 neurons: 4 states and a single torque input. The output has four layers for the time derivatives of the states. The hidden layers have 32 and 10 neurons. 

Learning rate is initialized as 0.02 and after each epoch it is reduced by 10%. Each batch consists of 32 seconds and 64 initial points. Each epoch is 1000 iterations and there are 50 epochs. We save the modle parameters after each epoch so a manual early stopping is implemented. 

#### 2.2.4.2. Dubins Car

One other system that is used in testing is the Dubins Car. The dynamics of the system is given in the implemented paper as
```math
\begin{bmatrix} 
\dot{x}_t & \dot{y}_t & \dot{\psi}_t
\end{bmatrix}^\top = 
\begin{bmatrix} 
v_t \cos(\psi_t) & v_t \sin(\psi_t) & \alpha_t \cdot \frac{v_t}{r} 
\end{bmatrix}^\top
```
where $(x_t, y_t)$ denotes the position of the robot, and $\psi_t$ denotes the heading angle. There are two control inputs, the linear velocity $v_t \in [0, v_{max}]$ and the steering $\alpha_t \in [-1, 1]$ [1]. Here, the parameters characterizing the system $v_{max}$ and $r$ are not given. Hence, they are chosen arbitrarily as equal to 1 while doing experiments on this system. 

The sampling frequency of the generated true data points are also not given. We used a sampling frequency of 80 Hz (i.e. a second is divided into 80 equally-distant time instants). The chosen time horizon for the trajectories, the method of the ODE solver, number of epochs and iterations, the amount of momentum used during training and batch sizes are also not given in the paper. These parameters also will be determined by experimentation. Finally, the specific number of neurons in the hidden layer of the three layer MLP structure are also not given in the paper, which are to be optimized by conducting experiments.

# 3. Experiments and results

## 3.1. Experimental setup

### 3.1.1. Training and Testing of Acrobot System Identification

As explained in the previous section, Acrobot is trained using random samples from the space. Then, a previously unknown trajectory is tested by giving the neural network to an ODE solver. Here, we first detail the network training procedure and provide various results.

System identification network is made up of a three layer MLP with sine activations. Hidden layers have 32 and 10 neurons. Loss function of this network is the ordinary mean square error (MSE) of the predicted $f_{\theta}$, just as in the paper. We have set the hyperparameters by trial-and-error and this particular experiment is conducted with the following parameters:

  - Batch time = 32 s
  - Batch size = 64 starting points
  - Dataset time = 128 s
  - Dataset size = 8192 data points
  - Iterations per epoch = 1000
  - Epochs = 15
  - Test frequency = every 20 iterations
  - Initial learning rate = 0.02
  - Learning rate decay rate = 10%
  - Optimizer = Adam
  - Weights initialization = Xavier uniform initialization with zero bias

The first plot Figure 2 shows the average loss per epoch.

<p align="center">
  <img src="/../main/AltunkolOzcan/images/loss-during-training.png" alt="Average loss at each epoch">
</p>
<p align="center">Figure 2: Average loss at each epoch</p>
<p align="center">
  <img src="/../main/AltunkolOzcan/images/loss-during-testing.png" alt="Loss over the whole training data every 20 iterations">
</p>
<p align="center">Figure 3: Loss over the whole training data every 20 iterations</p>

The training data seems promising but the loss is over the predicted state derivatives. when a state trajectory prediction is made using the learned dynamics, the error is actually integrated. In previously unseen test data as in Figure 4 and 5, we see the poor performance of the network.

<p align="center">
  <img src="/../main/AltunkolOzcan/images/test1.png" alt="A random trajectory test">
</p>
<p align="center">Figure 4: A random trajectory test</p>

<p align="center">
  <img src="/../main/AltunkolOzcan/images/test2.png" alt="A random trajectory test">
</p>
<p align="center">Figure 5: A random trajectory test</p>

At this stage, we have saved the network weights. We will proceed with more training using the saved weights with different hyperparameters.

Following our failed trials, we decided to increase network capacity. The new network width is set to 100. Batch size is set to 128. The results were 
### 3.1.2. Dubins Car

#### 3.1.2.1. Training and Testing of Dubins Car Trajectories

To reduce the complexity of debugging and the training procedure, a step-by-step method is followed to do a system identification of a Dubins Car system. Firstly, we experimented with the example code for learning a dynamical system using NeuralODEs (from the original implementation repo of NeuralODE paper [6]). By modifying this code, we created a training script to train a neural network to learn a specific Dubins Car trajectory. Using the assumed parameters and the data generation procedure explained in 2.4.4.2, a Dubins Car system trajectory is generated for a specified initial condition, time horizon, a random input vector, and a sampling frequency; by solving the system ODE, using Euler's method implemented inside the ```odeint``` function. This true trajectory is then used for obtaining batches of smaller trajectory sections, which consists of a group of trajectories starting from random time instants and continues for a specific duration. These batches are used to train the neural network by forward and backward passes at each iteration, inside an iteration loop. After a specific number of iterations, the whole trajectory is constructed using the network being trained and the average of absolute value differences between the true and predicted trajectories are printed to the screen. The trajectories also printed on the screen using matplotlib functions, in time domain and phase plane configurations.

The system identification section of the implemented paper does not actually use ODE solvers and NeuralODEs; instead it trains the MLP to learn the system dynamics by learning the derivatives of the system states. While training the controller, however, the ODE solvers are used to evaluate trajectories. The Dubins Car is a reasonably simple system, and training multiple neural networks simultaneously for the control system training part would be difficult. Hence, as an intermediate step (and as an experiment to exercise the methods), Dubins Car system identification experiments aimed to learn the system trajectories by calculating the loss function directly from the predicted and true trajectories, using $L_1$ loss. Many experiments are done by changing all of the hyperparameters and network structure, including activation functions. Using ReLU activations sometimes detrimentally affect the network performance, which is realized by visualising the activations and observing that all of the neurons die in some cases. Since the paper proposed sine as the activation, We also used sine instead of hyperbolic tangent or some other activation function. As the learning stops after some iterations for constant learning rates, we use an exponentially decaying learning rate. Here is the summary of the parameters of the learning procedure for learning a single trajectory of Dubins Car:

  - Batch time = 100 s
  - Batch size = 20 starting points
  - Dataset time horizon = 800 s
  - Dataset size = 64000 data points
  - Iterations per epoch = 200
  - Epochs = 20
  - Test frequency = every 20 iterations
  - Initial learning rate = 0.001
  - Learning rate decay rate = 10%
  - Optimizer = Adam
  - Number of neurons in the MLP layers: 5 - 50 - 20 - 3
  - Weights initialization = Random with zero mean and 0.1 standard deviation, with zero bias

After the system successfully learned a specific trajectory, the model parameters are exported and imported again to train the model with a different random trajectory. This procedure is done multiple times; however, the training performance does not seem to increase for small number of trajectories. We believe that we need to implement a single learning procedure to train using a larger set of trajectories, so that the system dynamics is captured by the network more robustly. Some examplary results of the last experiments (i.e. training the pre-trained model to predict other random trajectories) are given below.

<p align="center">
  <img src="/../main/AltunkolOzcan/images/epoch2_iter100.PNG" alt="Test of the predicted trajectory">
</p>
<p align="center">Figure 6: True and predicted trajectories in the beginning of the training</p>

<p align="center">
  <img src="/../main/AltunkolOzcan/images/epoch19_iter40.PNG" alt="Test of the predicted trajectory">
</p>
<p align="center">Figure 7: True and predicted trajectories at the end of the training</p>

<p align="center">
  <img src="/../main/AltunkolOzcan/images/loss_per_epoch.PNG" alt="Loss per epoch">
</p>
<p align="center">Figure 8: Loss value at the end of each epoch</p>

#### 3.1.2.2. Learning the Next State vs. Learning the State Transition Function

Until this point, we investigated the training procedure by constructing a neural network that learns the next state of the dubins car system, when the current states and control inputs are given as inputs. Even though this methodology successfully fits the given trajectories after some learning iterations, when another random trajectory is produced, the network prediction starts to fail significantly. By tuning the learning rate, batch size, and other such parameters we tried to train the network for thousands of epochs consisting of hundreds of iterations. However, we realized that the network does not converge to a successful one, and training stops after the average loss reaches around a certain value. In fact, the implemented paper trains the system identification network by making it learn the state transition function (i.e. the derivatives of the states), which may be more effective while dealing with system identification problems for several reasons. One possible reason might be that the derivative terms include some more information about the past and the future of the states depending on the system; however, we thought that with proper training, the system could still extract the same information and predict the next states, if it can predict the derivatives in another case. This brings other considerations, such as the fact that the states are not normalized values, which may bring some numerically unstable gradients when predicting the next state. Predicting derivatives by calculating the loss over the state transition function might be mitigating this effect in some way. However, in real-life cases, sampling states and taking derivatives in discrete systems may bring multiple problems, such as high distortion due to amplified noise. Still, this approach might be beneficial in some applications.



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

[5] Matthew Kelly (2024). Acrobot Derivation and Simulation (https://github.com/MatthewPeterKelly/Acrobot_Derivation_Matlab), GitHub. Retrieved December 10, 2024.

[6] D. Onken, L. Nurbekyan, X. Li, S. W. Fung, S. Osher, and L. Ruthotto, "A Neural Network Approach for High-Dimensional Optimal Control Applied to Multiagent Path Finding," IEEE Transactions on Control Systems Technology, 2022, doi: 10.1109/TCST.2022.3172872.
# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
