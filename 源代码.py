# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:29:39 2023

@author: 20051009 李明睿
多行注释内的内容为代码测试用
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

#————————————————————————————————————————画图部分——————————————————————————————————————————————


file=input("请输入文件名：")

nodes_num=int(input("请输入节点数："))
pole_num=int(input("请输入杆的数："))

data = pd.read_excel(file)
#print(data)


nodes_x = data.iloc[0:nodes_num,[0]].values
nodes_y = data.iloc[0:nodes_num,[1]].values



force_x=data.iloc[0:nodes_num,[4]].values
force_y=data.iloc[0:nodes_num,[5]].values

# 将force_x和force_y沿着列方向拼接成一个二维数组
force = np.concatenate((force_x, force_y), axis=1)

# 重塑数组，将x分量和y分量交错出现
force = force.reshape((1, -1)).T



fig = plt.figure()

#画节点
for i in range(nodes_num):
    if force_x[i][0]!=0 or force_y[i][0]!=0:
        plt.plot(nodes_x[i][0],nodes_y[i][0], marker='*', color='red')
    else:
        plt.plot(nodes_x[i][0],nodes_y[i][0], marker='.', color='black')
    
pole_node1=data.iloc[0:pole_num,[2]].values
pole_node2=data.iloc[0:pole_num,[3]].values

#画杆函数  
def draw_line(node1_x,node1_y,node2_x,node2_y):
    x = [[node1_x,node2_x]] 
    y = [[node1_y,node2_y]]

    for i in range(1):
       
        plt.plot(x[i], y[i], color='black')
#画杆
for j in range(pole_num): 
    draw_line(nodes_x[pole_node1[j][0]-1][0],nodes_y[pole_node1[j][0]-1][0],
              nodes_x[pole_node2[j][0]-1][0],nodes_y[pole_node2[j][0]-1][0])
    
plt.savefig("fig.png")


#————————————————————————————————————————求刚度矩阵———————————————————————————————————————————

#读取数据代码
#弧度制为输入参数
def sin(a):
    return np.sin(a)

def cos(a):
    return np.cos(a)

#定义坐标变换矩阵
def transfer_matrix(a):
    renmeda=np.array([[cos(a),sin(a),0,0],
                      [-sin(a),cos(a),0,0],
                      [0,0,cos(a),sin(a)],
                      [0,0,-sin(a),cos(a)]])
    return renmeda

#定义杆单元总体坐标系下的刚度矩阵,记得转成弧度制


#—————————————————————————————————————————定义节点类，杆类———————————————————————————————————————————————————————————
     


class Node():#节点类
    def __init__(self,num,x,y,force_x,force_y):
        self.num=num
        self.x=x
        self.y=y
        self.force_x=force_x
        self.force_y=force_y   


class pole():#杆类
    def __init__(self, beta,node1_num,node2_num,E,A,Node_list):
        
        
        #左右节点为杆的节点编号，左小右大,索引从0开始
        self.left_node_num=min(node1_num,node2_num)
        self.right_node_num=max(node1_num,node2_num)
        self.left_node=Node_list[min(node1_num,node2_num)-1]
        self.right_node=Node_list[max(node1_num,node2_num)-1]
        self.E=E
        self.A=A
        self.l=self.length()
        #杆的单元刚度矩阵
        self.beta=self.angle()
        self.K_e=self.calculate_K_e(E,A,self.l,self.beta)
        
        
        
    def length(self):
        #计算杆长
        x1, y1 = self.left_node.x, self.left_node.y
        x2, y2 = self.right_node.x, self.right_node.y
        dx, dy = x2 - x1, y2 - y1
        
        return (dx ** 2 + dy ** 2) ** 0.5     
    
    def angle(self):
        #计算杆的倾角
        x1, y1 = self.left_node.x, self.left_node.y
        x2, y2 = self.right_node.x, self.right_node.y
        dx, dy = x2 - x1, y2 - y1
        return math.atan2(dy, dx)#返回值为弧度制，可以直接用于sin，cos函数
    
        
    def calculate_K_e(self,E,A,l,alph):
        
        
        K_e=((E*A)/l)*np.array([[cos(alph)**2, cos(alph)*sin(alph), -cos(alph)**2, -cos(alph)*sin(alph)],
                     [cos(alph)*sin(alph), sin(alph)**2, -cos(alph)*sin(alph), -sin(alph)**2],
                     [-cos(alph)**2, -cos(alph)*sin(alph), cos(alph)**2, cos(alph)*sin(alph)],
                     [ -cos(alph)*sin(alph),  -sin(alph)**2, cos(alph)*sin(alph), sin(alph)**2]])
        
        return K_e
        

#-----------------------------------------------------------------------------------------------------

#导入节点
Nodes = []
for i in range(nodes_num):
   node = Node(f"Nodes {i}",x=nodes_x[i][0],y=nodes_y[i][0],force_x=force_x[i],force_y=force_y[i][0])
   Nodes.append(node)
   
E1 = float(input("请输入杆的杨氏模量："))
A1  = float(input("请输入杆的横截面积："))
    
#导入杆
Poles=[]
for j in range(pole_num):
    pole_i = pole(f"Poles {j}",node1_num=pole_node1[j][0],node2_num=pole_node2[j][0],E=E1,A=A1,Node_list=Nodes)
    Poles.append(pole_i)



#——————————————————————————————————————————总体刚度矩阵叠加——————————————————————————————————————————————————————
print("计算计算总体刚度矩阵")

#先建立总体刚度矩阵
K= np.zeros((2*nodes_num, 2*nodes_num))#平面行架结构一个节点两个自由度

for pole in Poles:
        # 找出杆两端节点的编号
        node1_num = pole.left_node_num
        node2_num = pole.right_node_num
       
        K_pole = pole.K_e
        # 叠加到总体刚度矩阵中
        K[2 * node1_num - 2:2 * node1_num, 2 * node1_num - 2:2 * node1_num] += K_pole[:2, :2]
        K[2 * node1_num - 2:2 * node1_num, 2 * node2_num - 2:2 * node2_num] += K_pole[:2, 2:]
        K[2 * node2_num - 2:2 * node2_num, 2 * node1_num - 2:2 * node1_num] += K_pole[2:, :2]
        K[2 * node2_num - 2:2 * node2_num, 2 * node2_num - 2:2 * node2_num] += K_pole[2:, 2:]
        
print("总体刚度矩阵计算完毕")


#——————————————————————————————————————边界条件——————————————————————————————————————————————————————————————————
print("引入边界条件")
#读取边界条件
boundary_x=data.iloc[0:nodes_num,[6]].values
boundary_y=data.iloc[0:nodes_num,[7]].values

boundary = np.concatenate((boundary_x, boundary_y), axis=1)

# 重塑数组，将x分量和y分量交错出现

boundary = boundary.reshape((1, -1)).T


def set_boundary_conditions(K, F, known_disp_nodes):
    """
    通过边界条件置1法来修改总体刚度矩阵
    KU=f

    Args:
        K: 总体刚度矩阵，形状为(n, n)
        F: 等式右侧的向量，形状为(n,)
        known_disp_nodes: 一个长度为n的布尔向量，表示每个节点的位移/位移变量是否已知

    Returns:
        K_mod: 修改后的总体刚度矩阵，形状为(n, n)
        F_mod: 修改后的等式右侧的向量，形状为(n,)
    """

    # 将对角线上相关节点的元素设为1，其他元素设为0
    K_mod = np.copy(K)
    for i in range(len(known_disp_nodes)):
        if known_disp_nodes[i][0]:
            K_mod[i, :] = 0
            K_mod[:, i] = 0
            K_mod[i, i] = 1

    # 调整等式右侧的向量
    F_mod = np.copy(F)
    for i in range(len(known_disp_nodes)):
        if known_disp_nodes[i][0]:
            F_mod [i][0] = 0

    return K_mod, F_mod


#经过边界条件处理后的刚度矩阵和载荷矩阵
K_mod,F_mod=set_boundary_conditions(K, force, boundary)


#------------------------------------------求解--------------------------------------------------------
#利用numpy的左除算法求解位移矩阵
u = np.linalg.solve(K_mod, F_mod)
u2= np.reshape(u, (nodes_num,2))

u_data = pd.DataFrame(u2,  columns=['x', 'y'])
u_data.to_csv('节点位移.csv')
print("计算完成")











