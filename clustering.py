import math
import numpy as np
import numpy.linalg as lin
import random as rand
import PriorityQueue as pq

def init_centroids(C, k, N):
    centroids = []
    for i in range(0, k):
        #Les representant sont choisis aleatoireent
        centroids.append(C[rand.randint(0, N-1)])
    return centroids

def euclidian_dist(x1, x2):
    s = 0
    for i in range(0, len(x1)):
        s += pow((x1[i]-x2[i]), 2)
    return math.sqrt(s)

def argmin(X, R):
    i_min = 0
    for i in range(R.shape[0]):
        if(euclidian_dist(X,R[i])<=euclidian_dist(X,R[i_min])):
            i_min = i
    return i_min

def sort_clust(C, R):
    pass
            
def kmeans(C, K, e):
    S = np.zeros((C.shape[0], K))
    assign = np.zeros(C.shape[0])
    R = init_centroids(C, K, len(R))
    b = True
    while(b):
        assign2 = np.copy(assign)
        S = np.zeros((C.shape[0], K))
        for i in range(0, C.shape[0]):
            assign[i] = argmin(C[i], R)
            S[i][assign[i]] = 1
        
        for k in range(0, K):
            n1=0        
            for i in range(0, C.shape[0]):        
                n1+=S[i][k]
            n2=np.zeros(C[0].shape[0])
            for i in range(0, C.shape[0]):        
                n2+=C[i]*S[i][k]
        b = e < (lin.norm(assign, assign2))
    return S

def violate_const(X, k, S, CL, ML):
    for i in range(0, ML.shape[0]):
        if ML[i][0] == X:
            if(S[ML[i][1]][k] == 0):
                return True
        if ML[i][1] == X:
            if(S[ML[i][0]][k] == 0):
                return True
        if CL[i][0] == X:
            if(S[CL[i][1]][k] == 1):
                return True
        if CL[i][1] == X:
            if(S[CL[i][0]][k] == 1):
                return True
        return False

def COPKmeans(C, K, e, CL, ML):
    S = np.zeros((C.shape[0], K))
    assign = np.zeros(C.shape[0])
    R = init_centroids(C, K, len(R))
    b = True
    while(b):
        assign2 = np.copy(assign)
        S = np.zeros((C.shape[0], K))
        for i in range(0, C.shape[0]):
            prio = sort_clust(C[i], R)
            b1 = True
            while(b1):
                k = prio.pop()
                assign[i] = k
                b1 = violate_const(C[i], k, S, CL, ML)
            S[i][assign[i]] = 1
        for k in range(0, K):
            n1=0        
            for i in range(0, C.shape[0]):        
                n1+=S[i][k]
            n2=np.zeros(C[0].shape[0])
            for i in range(0, C.shape[0]):        
                n2+=C[i]*S[i][k]
        b = e < (lin.norm(assign, assign2))
    return S
