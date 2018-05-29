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
    for i in range(len(R)):
        if(euclidian_dist(X,R[i])<=euclidian_dist(X,R[i_min])):
            i_min = i
    return i_min

def sort_clust(X, R):
    prio = pq.PriorityQueueMax()
    for i in range(len(R)):
        prio.push(i, euclidian_dist(X, R[i]))
    return prio
            
def kmeans(C, K, e):
    S = np.zeros((C.shape[0], K))
    assign = np.zeros(C.shape[0])
    R = init_centroids(C, K, len(C))
    b = True
    t=0
    while(b):
        t+=1
        assign2 = np.copy(assign)
        S = np.zeros((C.shape[0], K))
        for i in range(0, C.shape[0]):
            assign[i] = argmin(C[i], R)
            S[i][int(assign[i])] = 1
        
        for k in range(0, K):
            n1=0        
            for i in range(0, C.shape[0]):        
                n1+=S[i][k]
            n2=np.zeros(C[0].shape[0])
            for i in range(0, C.shape[0]):        
                n2+=C[i]*S[i][k]
        b = e < (lin.norm(assign-assign2))
    print t
    return assign

def violate_const(X, k, S, CL, ML):
    for i in range(0, len(ML)):
        if ML[i][0] == X:
            if(S[ML[i][1]][k] == 0):
                return True
        if ML[i][1] == X:
            if(S[ML[i][0]][k] == 0):
                return True
    for i in range(0, len(CL)):
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
    R = init_centroids(C, K, len(C))
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
                b1 = violate_const(i, k, S, CL, ML)
            S[i][int(assign[i])] = 1
        for k in range(0, K):
            n1=0        
            for i in range(0, C.shape[0]):        
                n1+=S[i][k]
            n2=np.zeros(C[0].shape[0])
            for i in range(0, C.shape[0]):        
                n2+=C[i]*S[i][k]
        b = e < (lin.norm(assign-assign2))
    return assign
