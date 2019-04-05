## About the Model DDPG (Deep Deterministic Policy Gradients)
This project implements an off-policy method called Multi agent Deep Deterministic Policy Gradient

### Background for DDPG
Multi Agent DDPG is an extended version of DDPG as described in the paper
[continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  > We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies end-to-end: directly from raw pixel inputs
  
Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

**DDPG** is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy, which is much easier to learn. DDPG also employs **Actor-Critic** model in which the Critic model learns the value function like DQN and uses it to determine how the Actor’s policy based model should change. The Actor brings the advantage of learning in continuous actions space without the need for extra layer of optimization procedures required in a value based function while the Critic supplies the Actor with knowledge of the performance.

For more information please visit [here](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) and [spinningup](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

To mitigate the challenge of unstable learning, a number of techniques are applied like Gradient Clipping, Soft Target Update through twin local / target network and Replay Buffer.

#### Experience Replay
In general, training and evaluating your policy and/or value function with thousands of temporally-correlated simulated trajectories leads to the introduction of enormous amounts of variance in your approximation of the true Q-function (the critic). The TD error signal is excellent at compounding the variance introduced by your bad predictions over time. It is highly suggested to use a replay buffer to store the experiences of the agent during training, and then randomly sample experiences to use for learning in order to break up the temporal correlations within different training episodes. This technique is known as experience replay. DDPG uses this feature
#### OU Noise
DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the agent were to explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their actions at training time. The authors of the original DDPG paper recommended time-correlated OU noise, but more recent results suggest that uncorrelated, mean-zero Gaussian noise works perfectly well. Since the latter is simpler, it is preferred. To facilitate getting higher-quality training data, you may reduce the scale of the noise over the course of training. 

please find the pseudocode from [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

### Multi Agent Deep Deterministic Policy Gradient
For this project I have used a variant of DDPG called Multi Agent Deep Deterministic Policy Gradient (MADDPG) which is described in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
  > We explore deep reinforcement learning methods for multi-agent domains. We begin by analyzing the difficulty of traditional algorithms in the multi-agent case: Q-learning is challenged by an inherent non-stationarity of the environment, while policy gradient suffers from a variance that increases as the number of agents grows. We then present an adaptation of actor-critic methods that considers action policies of other agents and is able to successfully learn policies that require complex multi-agent coordination. Additionally, we introduce a training regimen utilizing an ensemble of policies for each agent that leads to more robust multi-agent policies. We show the strength of our approach compared to existing methods in cooperative as well as competitive scenarios, where agent populations are able to discover various physical and informational coordination strategies.
  
### Code implementation
The code was implemented as a reference from the [Udacity DDPG Bipedel] (https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) and has been modified for the Tennis environment

The code consist of 
* `model.py` - Implement the Actor and the Critic classes.
  - The Actor and Critic classes each implements a Target and a Local Neural Networks used for the training.
* `multiagent_ddpg.py` - Implements the Multi-agent DDPG algorithm
  - The `maddpg` belongs to ddpg_agent
  - The helper function saving the checkpoints
  - providing `step()` and `act()` methods
  - The Multi-Agent Actor Critic `learn()` function slightly differs from the DDPG one, a `maddpg_learn()` method is provided here.
  - The `learn()` method updates the policy and value parameters using given batch of experience tuples.
* `ddpg_agent.py` - Implement the DDPG agent.
  - The Actor's Local and Target neural networks, and the Critic's Local and Target neural networks are instanciated by the Agent's constructor
  - The `learn()` method updates the policy and value parameters using given batch of experience tuples.
* `memory.py` - Implementation of Replay Buffer Memory
  - As it is accessed by both Agents, it is instanciated in the maddpg class instead of the ddpg class.
* `hyperparameters.py`  - Defines all the hyperparameters in constant variables.
* `utils.py`  -  Implement some helper functions to encode the states and actions before being inserted in the Replay Buffer, and decode them when a batch of experience is sampled from the Replay Buffer 
* `Tennis.ipynb` - This Jupyter notebooks allows to instanciate and train the agent

## Multi Agent DDPG implementation Observations and reasons

As a starting point, I mainly used the vanilla DDPG architecture and parameters used in the previous Udacity project/Tutorials (Neural networks with 2 dense layers of size 400 and 300, Learning rate of 1e-4 for the Actors and 1e-3 for the Critics networks, discount factor of 0.99, tau with value 1e-3)

Factors which made to converge faster are 
* Use a use normal distribution to sample experiences from the Replay Buffer
* Adding Batch Normalization after the Activation in the first layers of the neural network helped to converge a bit faster 
* Altering the Critics neural network so that the actions and states are concatenated directly at the input of the network.
* Changing the learning rates - Having similar and slightly higher learning rate for both the actor and the critic network and it helped solving the environment.  1e-4 (actors) and 5e-3 (critics) and batch size of 200
* Changing the units in the architecture doesn't made any difference.

**Actor Neural Network Architecture**
```
Input nodes (8x3 = 24)
  -> Fully connected nodes (400 nodes, Relu activation)
    -> Batch Normalization
      -> Fully Connected Layer (300 nodes, Relu activation)
        -> Ouput nodes (2 units/action, tanh activation)
```

**Critic Neural Network Architecture**
```
Input nodes ([ 8x3=24 states + 2 actions ] x 2 Agents = 52) 
  -> Fully Connected Layer (400 nodes, Relu activation)
    -> Batch Normlization
      -> Fully Connected Layer (300 nodes, Relu activation)
        -> Ouput node (1 node, no activation)
```

### Results

<img src="https://github.com/karthikrajkumar/collaboration-and-competition/blob/master/images/results.JPG" data-canonical-src="https://github.com/karthikrajkumar/collaboration-and-competition/blob/master/images/results.JPG" width="450" height="300" />


<img src="https://github.com/karthikrajkumar/collaboration-and-competition/blob/master/images/results%20graphical.JPG" data-canonical-src="https://github.com/karthikrajkumar/collaboration-and-competition/blob/master/images/results%20graphical.JPG" width="450" height="300" />

## Ideas for the future work
* Prioritized Experience Replay - To improve the Multi agent's performance, would be to implement the [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
  > Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. We use prioritized experience replay in Deep Q-Networks (DQN), a reinforcement learning algorithm that achieved human-level performance across many Atari games. DQN with prioritized experience replay achieves a new state-of-the-art, outperforming DQN with uniform replay on 41 out of 49 games.
* Try and implementing the various algorithms like MA-D4PG [D4PG](https://openreview.net/forum?id=SyZipzbCb&noteId=SyZipzbCb)
* As provided in the Open AI's Spinning up website, we can try to improve the performance by implementing the Twin Delayed DDPG ([TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html))
  > While DDPG can achieve great performance sometimes, it is frequently brittle with respect to hyperparameters and other kinds of tuning. A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function. Twin Delayed DDPG (TD3) is an algorithm which addresses this issue by introducing three critical tricks:
  * Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.
  * Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.
  * Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.
 Together, these three tricks result in substantially improved performance over baseline DDPG



