[Bootstrap-based Causal Structure Learning](https://dl.acm.org/doi/abs/10.1145/3511808.3557249) <br>

# Usage
"BCSL.m" is main function. <br>
Note that the current code has only been debugged on Matlab (2018a) with a 64-bit Windows system and supports only discrete datasets.<br>
----------------------------------------------
function [DAG, time] = BCSL(Data, Alpha, rand_sample_numb) <br>
* INPUT: <br>
```Matlab
Data is the data matrix, and rows represent the number of samples and columns represent the number of nodes. If Data is a discrete dataset, the value in Data should start from 1.
Alpha is the significance level, e.g., 0.01 or 0.05.
rand_sample_numb is the number of sub-datasets generated, e.g., 15.
```
* OUTPUT: <br>
```Matlab
DAG is a directed acyclic graph learned on a given dataset。
time is the runtime of the algorithm.
```

# Example for discrete dataset
```Matlab
clear;
clc;
addpath(genpath('common_func/'));
alpha=0.01;
data=load('./dataset/Alarm_1000s.txt');
data=data+1;
[DAG, time] = BCSL(data, alpha, 15);
```

# Reference
* Guo, Xianjie, et al. "Bootstrap-based Causal Structure Learning." *Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM'22)* (2022).
