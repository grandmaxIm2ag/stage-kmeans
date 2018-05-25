#!/usr/bin/python2.7
#-*- coding: utf-8 -*-

import heapq as hq


class PriorityQueueMax:

    def __init__(self):
        self.h = []
        self.size=0

    def push(self, elem, priority):
        hq.heappush(self.h, (-priority, elem))
        self.size+=1
        return
    
    def pop(self):
        self.size-=1
        return hq.heappop(self.h)[1]

    def top(self):
        [p, elem] = hq.heappop(self.h)
        hq.heappush(self.h, (p, elem))
        return elem

    def top_with_priority(self):
        [p, elem] = hq.heappop(self.h)
        hq.heappush(self.h, (p, elem))
        return (-p, elem)

    def isEmpty(self):
        return (self.size == 0)

    def get_size(self):
        return self.size
    
    def remove(self, elem):
        tmp = []
        first = True
        for i in range(self.size):
            p,e = hq.heappop(self.h)
            if e == elem and first:
                first = False
                continue
            hq.heappush(tmp, (p, e))
        self.h = tmp
        self.size-=1
