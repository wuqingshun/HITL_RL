# Adaptive Task Assignment in Spatial Crowdsourcing: A Human-in-the Loop Approach
## In recent years, adaptive task assignment has been explored in spatial crowdsourcing. The challenge lies in how to adaptively partition the task stream to achieve the best utility for task assignment. A number of existing works have attempted to solve this challenge and achieve better performance by utilizing learning-based methods. Specifically, they mainly employ reinforcement learning to divide the task stream into a series of suitable batches and then perform task assignment in a batch fashion. Drawing inspiration from the effectiveness of human-machine collaborative decision-making, we aim to investigate human-in-the-loop methods to further enhance the performance of adaptive task assignment. In this paper, we propose a novel framework called Human-in-the-Loop Adaptive Partition (HLAP), which consists of two primary modules: Reinforcement Learning Partition Decision (RL-PD) and Human Supervision and Guidance (HSG). In the RL-PD module, we develop an RL agent, referred to as the decision-maker, by integrating the dual attention network into the Deep Q-Network (DQN) algorithm to capture global contextual information and long-range dependencies for a better understanding of the environment. In the HSG module, we design a human-in-the-loop mechanism to optimize the performance of the decision-maker, focusing on addressing two key issues: when and how humans interact with the decision-maker. Furthermore, to alleviate the heavy workload on humans, we construct a supervisor based on RL to oversee the decision-maker's partition process and adaptively determine when human intervention is necessary. We conduct extensive experiments on two real-world datasets, and the results demonstrate the efficiency and effectiveness of the HLAP framework.
