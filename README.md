# Implements of Reinforcement Learning Algorithms

This repo is implements of Reinforcement Learning Algorithms, implementing as learning, some of them are even another version of some tutorial. Any contributions are welcomed.

# Content

+ [Deep Deterministic Policy Gradient (DDPG)](/algorithms/DDPG/ddpg.py)   
Implement of DDPG.
    > arXiv:1509.02971: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

+ [Asynchronous Advantage Actor-Critic Model (A3C)](/algorithms/A3C/a3c.py)   
Implement of A3C.
    > arXiv:1602.01783: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

+ [Double-DQN](/algorithms/Double-DQN/double-dqn.py)  
Implement of Double-DQN.
    > arXiv:1509.06461: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

+ [Dueling-DQN](/algorithms/Dueling-DQN/dueling-dqn.py)   
Implement of Dueling-DQN.
    > arXiv:1511.06581: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

+ [Deep Q-Network (DQN)](/algorithms/DQN/dqn.py)   
Implement of DQN.
    > arXiv:1312.5602: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

+ [Actor-Critic Model](/algorithms/Actor-Critic/a-c.py)    
Implement of Actor-Critic Model.
    > arXiv:1607.07086: [An Actor-Critic Algorithm for Sequence Prediction](https://arxiv.org/abs/1607.07086)

+ [Policy Gradient (PG)](/algorithms/Policy-Gradient/pg.py)   
Implement of Policy Gradient.
    > NIPS. Vol. 99. 1999: [Policy gradient methods for reinforcement learning with function approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

+ [Q-Learning](/algorithms/Q-Learning/q.py)   
Implements of Q-Learning.
    > Paper: [Convergence of Q-learning: a simple proof](http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf)

+ [Sarsa](/algorithms/Sarsa/sarsa.py)   
Implement of Sarsa.
    > Paper: [Online Q-Learning using Connectionist Systems](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf)
    
# Requirements

- Python3.5
- TensorFlow1.4
- gym
- numpy
- matplotlib
- pandas (option)

# How to Run

All algorithms are implemented with TensorFlow, the default environment are games provided by gym. You can just clone this project, and run the each algorithm by:
```
python3.5 algorithms/algo_name.py
```

# TODO

- More implements of Deep Reinforcement Learning Paper and Methods.