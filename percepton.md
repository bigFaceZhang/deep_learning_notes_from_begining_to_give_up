1. **感知机**

- 特点  
```
二分类  
线性分类器
```
- 模型 
```math
f(x) = sign(\omega \cdot x + b)
```
- 损失函数
```math
min_{\omega, b}L(\omega, b) = -\sum_{x_i \in M}y_i(\omega \cdot x_i + b)
```  

[comment]: <> (半方大的空白&ensp;或&#8194;)  
[comment]: <> (全方大的空白&emsp;或&#8195;)  
[comment]: <> (不断行的空白格&nbsp;或&#160;)    

[comment]: <> (This is a comment, it will not be included)
[comment]: <> (in  the output file unless you use it in)  
[comment]: <> (a reference style link.)  
[//]: <> (This is also a comment.)  
[//]:"Comment"   
&ensp;&ensp;&ensp;&ensp;其中，M为分类错误数据点集合。  

- 梯度下降
```math
\triangledown_w L(\omega, b) = -\sum_{x_i \in M}y_ix_i

\triangledown_b L(\omega, b) = -\sum_{x_i \in M}y_i
```  

&ensp;&ensp;&ensp;&ensp;每次可以取一个样本计算梯度（随机梯度下降），也可取多个样本（批量梯度下降）求平均梯度。  
&emsp;&emsp;学习率`$\eta$`（learning rate，`$0 < \eta \leq 1$`），也被称作步长，即梯度在每次迭代时只更新一小部分。  
&emsp;&emsp;注：参数是往梯度的负方向上更新。
```math
\omega \leftarrow \omega - (-\eta y_ix_i)  

b \leftarrow b - (-\eta y_i)
```
- 算法过程  
> 输入：  
> &emsp;&emsp;训练集：`$D = \{(x_1,x_1), (x_2,x_2), \cdots (x_N,x_N)\}$`，其中，`$x_i\in R^n$`，`$y_i \in \{-1, +1\}$`, i = 1, 2, 3, `$\cdots$` N；  
> &emsp;&emsp;学习率：`$\eta$`（`$0 < \eta \leq 1$`）；  
> 输出：`$\omega, b$`；`$f(x) = sign(\omega \cdot x + b)$`；  
> (1) 选取初值`$\omega_0, b_0$`, 一般赋值为比较小接近0的随机数；  
> (2) 在训练集中选取数据`$x_i,y_i$`，可以选择1个或者多个；  
> (3) 计算`$y_i(\omega \cdot x_i + b)$`;  
> (4) 计算梯度并更新：
> ```math
> \omega \leftarrow \omega + \eta y_ix_i  
> 
> b \leftarrow b + \eta y_i
> ```  
> (5) 重复(2)-(4)，直至训练集中没有分类错误点或达到停止条件；
