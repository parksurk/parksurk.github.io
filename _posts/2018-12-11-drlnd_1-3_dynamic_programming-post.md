---
title: "DRL Monte Carlo Mothods"
excerpt: "심층강화학습(Deep Reinforcement Learning) Dynamic Programming 알고리즘에 대해 알아보자."
date: 2018-12-10 00:00:01 -0400
categories:
  - DeepRL
tags:
  - DeepRL
---


```python
from IPython.display import Image
Image(filename='./images/1-0-0-1_opening.jpg')
```




![jpeg](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_0_0.jpeg)



# Lesson 1-3: Dynamic Programming

In the **dynamic programming** setting, the agent has full knowledge of the Markov decision process (MDP) that characterizes the environment. (This is much easier than the **reinforcement learning** setting, where the agent initially knows nothing about how the environment decides state and reward and must learn entirely from interaction how to select actions.)

This lesson covers material in **Chapter 4 (especially 4.1-4.4) of the Sutton's textbook**.

## 1-3-1 : Grid Example


```python
from IPython.display import Image
Image(filename='./images/1-3-1-1_grid_example.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_3_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-1-2_grid_example.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_4_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-1-3_grid_example.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_5_0.png)



## 1-3-2 : Iterative Method


```python
from IPython.display import Image
Image(filename='./images/1-3-2-1_dp_iterative_method_step1.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_7_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-2_dp_iterative_method_step2.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_8_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-3_dp_iterative_method_step3.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_9_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-4_dp_iterative_method_step4.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_10_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-5_dp_iterative_method_step5.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_11_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-6_dp_iterative_method_step6.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_12_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-7_dp_iterative_method_step7.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_13_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-8_dp_iterative_method_step8.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_14_0.png)



## Notes on the Bellman Expectation Equation

In the previous example, we derived one equation for each environment state. For instance, for state s1 we saw that:


```python
from IPython.display import Image
Image(filename='./images/1-3-2-9_dp_iterative_method_Notes_on_the_Bellman_Expectation_Equation.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_16_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-10_dp_iterative_method_Notes_on_the_Bellman_Expectation_Equation.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_17_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-11_dp_iterative_method_Notes_on_the_Bellman_Expectation_Equation.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_18_0.png)



## Notes on Solving the System of Equations


```python
from IPython.display import Image
Image(filename='./images/1-3-2-12_dp_iterative_method_Notes_on_Solving_the_System_of_Equations.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_20_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-13_dp_iterative_method_Notes_on_Solving_the_System_of_Equations.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_21_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-2-14_dp_iterative_method_Notes_on_Solving_the_System_of_Equations.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_22_0.png)



Note : This example serves to illustrate the fact that it is possible to directly solve the system of equations given by the Bellman expectation equation for vπ. However, in practice, and especially for much larger Markov decision processes (MDPs), we will instead use an iterative solution approach.

## 1-3-3. Iterative Policy Evaluation


```python
from IPython.display import Image
Image(filename='./images/1-3-3-1_dp_Iterative-Policy-Evaluation_step1.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_25_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-3-2_dp_Iterative-Policy-Evaluation_step2.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_26_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-3-3_dp_Iterative-Policy-Evaluation_step3.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_27_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-3-4_dp_Iterative-Policy-Evaluation_step4.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_28_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-3-5_dp_Iterative-Policy-Evaluation_step5.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_29_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-3-6_dp_Iterative-Policy-Evaluation_step6.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_30_0.png)



## 1-3-4 : Policy Improvement


```python
from IPython.display import Image
Image(filename='./images/1-3-4-1_dp_policy_evaluation_step1.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_32_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-2_dp_policy_evaluation_step2.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_33_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-3_dp_policy_evaluation_step3.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_34_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-4_dp_policy_evaluation_step4.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_35_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-5_dp_policy_evaluation_step5.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_36_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-6_dp_policy_evaluation_step6.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_37_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-7_dp_policy_evaluation_step7.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_38_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-8_dp_policy_evaluation_step8.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_39_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-9_dp_policy_evaluation_step9.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_40_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-10_dp_policy_evaluation_step10.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_41_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-11_dp_policy_evaluation_step11.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_42_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-12_dp_policy_evaluation_step12.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_43_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-13_dp_policy_evaluation_step13.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_44_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-14_dp_policy_evaluation_step14.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_45_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-15_dp_policy_evaluation_step15.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_46_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-4-16_dp_policy_evaluation_step16.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_47_0.png)



## 1-3-5 : Policy Iteration


```python
from IPython.display import Image
Image(filename='./images/1-3-5-1_dp_Policy-Iteration_step1.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_49_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-5-2_dp_Policy-Iteration_step2.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_50_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-5-3_dp_Policy-Iteration_step3.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_51_0.png)



## 1-3-6 : Truncated Policy Iteration


```python
from IPython.display import Image
Image(filename='./images/1-3-6-1_dp_Truncated-Policy-Iteration_step1.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_53_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-6-2_dp_Truncated-Policy-Iteration_step2.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_54_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-6-3_dp_Truncated-Policy-Iteration_step3.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_55_0.png)



## 1-3-7 : Value Iteration


```python
from IPython.display import Image
Image(filename='./images/1-3-7-1_dp_Value-Iteration_step1.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_57_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-7-2_dp_Value-Iteration_step2.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_58_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-7-3_dp_Value-Iteration_step3.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_59_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-7-4_dp_Value-Iteration_step4.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_60_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-7-5_dp_Value-Iteration_step5.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_61_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-7-6_dp_Value-Iteration_step6.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_62_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-7-7_dp_Value-Iteration_step7.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_63_0.png)




```python
from IPython.display import Image
Image(filename='./images/1-3-7-8_dp_Value-Iteration_step8.png')
```




![png](/assets/images/2018-12-11-drlnd_1-3_dynamic_programming-post/output_64_0.png)



## 1-3-8 : Quiz : Check Your Understanding

Match each algorithm to its appropriate description.

1. Value Iteration
2. Policy Improvement
3. Policy Iteration
4. (Iterative) Policy Evaluation

### QUESTION 1 OF 4

Finds the optimal policy through successive rounds of evaluation and improvement.

### QUESTION 2 OF 4

Given a value function corresponding to a policy, proposes a better (or equal) policy.

### QUESTION 3 OF 4

Computes the value function corresponding to an arbitrary policy.

### QUESTION 4 OF 4

Finds the optimal policy through successive rounds of evaluation and improvement (where the evaluation step is stopped after a single sweep through the state space).
