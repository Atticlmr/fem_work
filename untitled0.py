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
nodes_num=6
pole_num=10

file ="C:/Users/asus/Desktop/test_demo_1.xlsx"
data = pd.read_excel(file)
#print(data)


nodes_x = data.iloc[0:nodes_num,[0]].values
nodes_y = data.iloc[0:nodes_num,[1]].values

force_x=data.iloc[0:nodes_num,[4]].values
force_y=data.iloc[0:nodes_num,[5]].values
fig = plt.figure()
#画节点
for i in range(nodes_num):
    if force_x[i][0]!=0 or force_y[i][0]!=0:
        plt.plot(nodes_x[i][0],nodes_y[i][0], marker='+', color='red')
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


#————————————————————————————————————————求刚度矩阵———————————————————————————————————————————

#读取数据代码


abc=8#杆与x轴的夹角（角度制）
E=1#弹性模量
A=1#杆截面积
l=1#杆长

#先建立总体刚度矩阵
nodes_num=2#节点数
K= np.zeros((2*nodes_num, 2*nodes_num))#平面行架结构一个节点两个自由度


def sin(a):
    return math.sin(a)

def cos(a):
    return math.cos(a)

#定义坐标变换矩阵
def transfer_matrix(b):
    #角度制转弧度制
    a=b*math.pi/180
    #杆系的坐标变换矩阵
    renmeda=np.array([[cos(a),sin(a),0,0],
                      [-sin(a),cos(a),0,0],
                      [0,0,cos(a),sin(a)],
                      [0,0,-sin(a),cos(a)]])
    return renmeda

#定义杆单元总体坐标系下的刚度矩阵,记得转成弧度制
def K_e(E,A,l,alph):
    
    K_e=((E*A)/l)*np.array([[cos(alph)**2, cos(alph)*sin(alph), -cos(alph)**2, -cos(alph)*sin(alph)],
                 [cos(alph)*sin(alph), sin(alph)**2, -cos(alph)*sin(alph), -sin(alph)**2],
                 [-cos(alph)**2, -cos(alph)*sin(alph), cos(alph)**2, cos(alph)*sin(alph)],
                 [ -cos(alph)*sin(alph),  -sin(alph)**2, cos(alph)*sin(alph), sin(alph)**2]])
    
    return K_e

#————————————————————————————————————————————————————————————————————————————————————————————————————
class pole():
    def __init__(self, beta,node1,node2,E,A,l):
        
        self.beta=beta
        #左右节点为杆的节点编号，左小右大
        self.left_node=min(node1,node2)
        self.right_node=max(node1,node2)
        self.E=E
        self.A=A
        self.l=l
        #杆的单元刚度矩阵
        self.K_e=K_e(E,A,l,beta)


class Node():
    def __init__(self,num,x,y,force_x,force_y):
        Node.num=num
        Node.x=x
        Node.y=y
        Node.force_x=force_x
        Node.force_y=force_y   
        
        
#----------------------------------------------------------------------------------------------------


for i in range(6):
    locals()['Node_'+str(i)] = Node(i,nodes_x[i],nodes_y[i],force_x[i],force_y[i])
    
print(Node_0.x)

#总体刚度矩阵的叠加
'''
def sum_K_e():
    return 0

pole1=pole(beta=0,node1=4,node2=3,E=4,A=5,l=6)
'''









    











