# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:54:58 2024

@author: vabno
"""
import numpy as np

from numpy import(dot)
from numpy.linalg import norm

import matplotlib.pyplot as plt
import sympy

import math
import scipy.special


def indexPoint(rows, cols, x):
    #Takes a grid size and an index, returns coords of point on that grid
    #with the given index. Index is assigned from bottom left to top right in
    #the expected way with the first inidex being 0
   
    res = [(x)%cols, ((x) // cols)]
    if x < 0 or x > (rows*cols)-1:
        res = -1
    return res

def pointIndex(rows,cols, p):
    #Takes a grid size and a point on that grid and returns the index of that
    #point given the above index system
   
    res = p[0] + (cols * p[1])
    if res < 0 or res > (rows*cols)-1:
        res = -1
    return res


def nextTriangle(rows, cols, T):
    #Takes a triangle and outputs a different triangle that has its highest
    #index point moved one index higher, if at limit moved second highest one
    #higher and brings highest to one above that new value, if both are at the
    #highest index possible it moves the lowest index up and brings the other
    #two down so that they are just above it as well.
   
    pointA = pointIndex(rows,cols,T[0])
    pointB = pointIndex(rows,cols,T[1])
    pointC = pointIndex(rows,cols,T[2])
    inOrder = np.sort([pointA,pointB,pointC])
    if inOrder[2] < (rows * cols) - 1:
        inOrder[2] += 1
    elif inOrder[2] - inOrder[1] > 1:
        inOrder[1] += 1
        inOrder[2] = inOrder[1]+1
    elif inOrder[1] - inOrder[0] > 1:
        inOrder[0] += 1
        inOrder[1] = inOrder[0]+1
        inOrder[2] = inOrder[1]+1
    else:
        inOrder = [0,1,2]
    res = [indexPoint(rows,cols,inOrder[0]),
           indexPoint(rows,cols,inOrder[1]),
           indexPoint(rows,cols,inOrder[2])]
    return res
       
   
   
   


def triangleType(cosa,cosb,cosc):
    #Helper function for triangle test
    if cosa == 0 or cosb == 0 or cosc == 0:
        return "right"
    elif  cosa < 0 or  cosb < 0 or  cosc < 0:
        return "obtuse"
    else:
        return "acute"
       
def triangleTest(a, b, c, d, e, f):
    #returns string of type of triangle ((a,b), (c,d), (e,f))
    if isNonTriangle([[a,b],[c,d],[e,f]]):
        return "non"
    vectors = ([c-a, d-b],[a-e, b-f], [e-a, f-b], [c-e, d-f])
    u = vectors [0]
    v = vectors [1]  
    v1 = vectors [2]
    w1 = vectors [3]
    #print(u, v, w1)
    cos1 = dot(u,v1)
    cos2 = dot(u,w1)
    cos3 = dot(v,w1)
    #print(cos1, cos2, cos3)
    return triangleType(cos1, cos2, cos3)


def countTriangles(rows, cols, makeTable):
    #Returns tuple containinge the number of total, non, right, acute,
    #and obtuse triangles
    #WARNING: NEEDS AT LEAST 3 ROWS OR IT GIVES RUNTIME ERROR
    res = 1
    t = [[0,0],[1,0], [2,0]]
    non = 0
    right = 0
    acute = 0
    obtuse = 0
    testResult = triangleTest(t[0][0], t[0][1],
                       t[1][0], t[1][1],
                       t[2][0], t[2][1])
    if testResult == "non":
        non +=1
    if testResult == "right":
        right +=1
    if testResult == "acute":
        acute +=1
    if testResult == "obtuse":
        obtuse +=1
    t = nextTriangle(rows,cols,t)
   
    while t != [[0,0], [1,0], [2,0]]:
        res+=1
        testResult = triangleTest(t[0][0], t[0][1],
                           t[1][0], t[1][1],
                           t[2][0], t[2][1])
        if testResult == "non":
            non +=1
        if testResult == "right":
            right +=1
        if testResult == "acute":
            acute +=1
        if testResult == "obtuse":
            obtuse +=1
       
        t = nextTriangle(rows,cols,t)
    if makeTable:
        print(f"{rows} by {cols} lattice \n non: {non},  right: {right}, acute: {acute}, obtuse: {obtuse} , total : {res}")
    return res, non, right, acute, obtuse
   


def isNonTriangle(T):
    #helper functioin for triangleTest, checks for non-triangles in a way that
    #avoids floating point errors
    if T[0][0] == T[1][0] == T[2][0]:
        return True
    if T[0][1] == T[1][1] == T[2][1]:
        return True
    a = T[0][0] - T[1][0]
    b = T[0][1] - T[1][1]
    c = T[0][0] - T[2][0]
    d = T[0][1] - T[2][1]
    if a*d == c*b:
        return True
    else:
        return False
   
   
def plotLimit(start, stepSize, samples, rows):
    #Plots probability of obtuse triangles up to given limit
    #WARNING: NEEDS TO START AT AT LEAST 3 ROWS
    i = 0
    x = start
    L = []
    v = 1 - (1/(rows*rows))
    M = [v]*(samples)
    while i < samples:
        types = countTriangles(rows,x, True)
        prob = types[4]/types[0]
        L.append(prob)
     
        x+=stepSize
        i+=1
       
    xAxis = np.arange(start, start+samples*stepSize, stepSize)
   

    plt.plot(xAxis, M, c='red')
    plt.scatter(xAxis,L, c='#1f77b4')
   
    plt.xlabel("n")  
    plt.ylabel("P(OB)")
    plt.title(f"Probability of an Obtuse Triangle, m = {rows}")
    plt.yticks(np.linspace(0,1,11))
   
   
   
       

   
    plt.show()
    return
   
   
'''
plotLimit(3,1,48,2)
plotLimit(3,1,48,3)
plotLimit(3,1,48,4)
plotLimit(3,1,48,5)
plotLimit(3,1,48,6)
plotLimit(3,1,48,7)
plotLimit(3,1,48,8)
'''


def kCellCoords(kT, rows, cols):
    res = []
    return res


def primeRun(rows, start, end):
    cols = start
    while cols <= end:
        non = NT2(rows,cols)
        if sympy.isprime(non):
            print (non)
        cols +=1
    return
   


def comb (n,k):
    if k < 0 or k > n:
        return 0
    num = 1
    den = 1
    i = 0
    while i < k:
        num *= n - i
        den *= 1+i
        i+=1
    return   num//den      
   
       

def NT(rows,cols):
    if rows > cols :
        (rows,cols) = (cols,rows)
    return DNT(rows,cols) + rows * comb(cols,3) + cols * comb(rows,3)





def DNT(rows, cols):
    if rows < 3: return 0
    if cols < 3: return 0
    rec = 2*DNT(rows-1, cols) - DNT(rows-2, cols)
    i = 2
    res = 0
    while i <= cols:
        summand = 2 * (math.gcd(i-1,rows-1)-1) * (cols-i+1)
        res += summand
        i+=1
    return rec+res

def DNT2(rows,cols):
    i = 2
    total = 0
    while i <= rows:
        k = 2
        while k <= cols:
            s = 2 * (math.gcd(k-1,i-1)-1) * (rows - i + 1) * (cols - k + 1)
            total += s
            k+=1
        i += 1
    return total
       
def NT2(rows,cols):
    if rows > cols :
        (rows,cols) = (cols,rows)
    return DNT2(rows,cols) + rows * comb(cols,3) + cols * comb(rows,3)



def a(i,m,k):
    if m <=1 or k <=1 : return 0
    res = (i-1)/(m-1)
    res *= (k-i)
    res -= 1
    res =  np.ceil(res)
    res = max(res,0)
    return min(res,m-2)

def b(i,m,k):
    if m <=1 or k <=1 : return 0
    res = ((i-1)/2)**2 + ((m-1)/2)**2 - ((k-1)-((i-1)/2))**2
    if res <= 0: return 0
    x = np.sqrt(res)
    if x <= 0:  return 0
    if m//2 == (m-1)//2:
        return 2 * np.ceil(x-1) + 1
    else:
        return 2 * np.ceil(x-0.5)
   
       
def oMK(m,k):
    res = 0
    i = 2
    while i < k:
        res += 4*a(i,m,k)
        res += 4*b(i,m,k)
        i+=1
    res += 2 * b(1,m,k)
    res += 2 * b(1,k,m)
    res += 2 * (m*k - 3 - np.gcd(m-1,k-1))
    return max(res,0)



def OB(m,k):
    res = 0
    for i in range (2,m+1):
        for j in range (2,k+1):
            res += oMK(i,j) * (m-i+1) * (k-j+1)
    return res

print(OB(10, 10))
