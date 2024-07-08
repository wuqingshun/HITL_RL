# Adaptive Task Assignment in Spatial Crowdsourcing: A Human-in-the Loop Approach

In this repository, we open source a framework called Human-in-the-Loop Adaptive Partition (HLAP), which consists of two primary modules: Reinforcement Learning Partition Decision (RL-PD) and Human Supervision and Guidance (HSG). In the RL-PD module, we develop an RL agent, referred to as the decision-maker, by integrating the dual attention network into the Deep Q-Network (DQN) algorithm to capture global contextual information and long-range dependencies for a better understanding of the environment. In the HSG module, we design a human-in-the-loop mechanism to optimize the performance of the decision-maker, focusing on addressing two key issues: when and how humans interact with the decision-maker. Furthermore, to alleviate the heavy workload on humans, we construct a supervisor based on RL to oversee the decision-maker's partition process and adaptively determine when human intervention is necessary. 

# Requirements
```
* PyTorch >= 1.0.0
* Python 3.6+
* Conda (suggested for building environment etc)
* tensorboardx==1.9
* tensorflow==1.14.0 (non-gpu version will do, only needed for tensorboard)
* matplotlib==3.3.4 （draw training results）
```

# Project Structure

```
-dataset/
    |-map_* (chengdu or haikou road network data)
    |-task_*_0_24.txt (training and testing data for a particular algorithm)
-src/
    |-config_*.py (config files of a particular algorithm)
    |-model_*.py (model definition for a particular algorithm)
    |-memory.py (action replay memory buffer)
-main/
    |-env_*.py (training and testing environment in Chengdu or Haikou)
    |-run_*.py (training file of a particular algorithm)
```

# How to run?

To run a particular algorithm (say DDAQN) one can do ``python run_DDAQN.py`` this will generate the trace for that algorithm.
