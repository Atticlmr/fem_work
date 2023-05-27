# 有限元大作业

## 输入方式：

excel的xlsx文件,文件需要和exe同处于一个文件夹下

![](C:\Users\lmr\AppData\Roaming\marktext\images\2023-05-25-10-52-53-image.png)

按照文件中的表头输入相应的数据即可，节点的数据需要按照节点编号顺序依次输入

nodes_x和nodes_y是节点的xy坐标

pole_node1和pole_node2是杆的两端的节点编号

force_x和force_y是施加在节点上的力的xy分量

位移约束这一项，对应节点**存在约束**则填1，**不**存在约束则填0，不能不填

![](C:\Users\lmr\AppData\Roaming\marktext\images\2023-05-25-10-56-12-image.png)

注意，输入单位要一致

双击fem_work.exe，等待几秒后

需要输入excel文件名，注：文件名不能包括空格，需要包括后缀。需要额外输入节点数和杆数，按照提示操作即可

![](C:\Users\lmr\AppData\Roaming\marktext\images\2023-05-25-10-56-54-image.png)

注：200e9即$200\times 10^9$ ,1e-2即$1\times 10^{-2}$ 

回车后程序开始运行,输出png文件和csv文件

## 

## 输出方式：

输出会在同一个文件夹下产生一个csv文件和png图片，名称如下

![](C:\Users\lmr\AppData\Roaming\marktext\images\2023-05-24-17-22-27-image.png)

png图片是桁架结构的样子，对于有施加外力的节点，会被标注红色“*”号

![](C:\Users\lmr\AppData\Roaming\marktext\images\2023-05-25-10-59-16-image.png)

csv文件是各个节点产生的位移，编号从0开始

![](C:\Users\lmr\AppData\Roaming\marktext\images\2023-05-25-10-59-38-image.png)

***打开csv文件时不要用excel*** ！！！***直接用记事本打开*** 

因为用excel打开时excel显示的数据会保留两位有效数字

注：如果出现数量级相较其他数值明显非常小的数，比如$1$ 和$10^{-9}$  ,$10^{-9}$  可视为0，这是python底层算法对浮点精度的优化的问题,会导致一些本应该为0的值变成一个特别小的数

## 

## 源代码所需的环境

如果需要运行源代码

则需要numpy、matplotlib、pandas、math库

python版本为3.8
