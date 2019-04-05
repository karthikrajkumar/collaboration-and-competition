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
