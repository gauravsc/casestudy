import math
import numpy as np
import sys

def dist_man((x1, y1), (x2, y2)):
    return abs(x1 - x2) + abs(y1 - y2)

def dist_euc((x1, y1), (x2, y2)):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def total_dist(nodes, cls, dist):
    tot_dist = 0
    for i in xrange(len(cls)):
        tot_dist += dist[nodes[int(cls[i])]][nodes[i]]

    return tot_dist

def assign_to_closest(nodes, meds, dist, cls = None):
    if cls == None:
        cls = np.empty(len(nodes))

    for i in xrange(len(nodes)):
        if i in meds:
            cls[i] = i
            continue
        d = sys.maxint
        for j in xrange(len(meds)):
            d_tmp = dist[nodes[i]][nodes[meds[j]]]
            if d_tmp < d:
                d = d_tmp
                cls[i] = meds[j]

    return cls

def clarans_basic(nodes, k, numlocal, minmaxneighbor=250, p=0.0125, dist=dist_euc):
    if p < 0 or p > 1:
        raise ValueError('Argument p has to be in range [0,1].')

    # Calculate the maxneighbor attribute
    n = len(nodes)
    if k * (n-k) <= minmaxneighbor:
        maxneighbor = k * (n-k)
    else:
        maxneighbor = minmaxneighbor + int(p * (k * (n - k)))

    print 'CLARANS[numlocal:%d,maxneighbor:%d] computing...' % (numlocal, maxneighbor)

    # Initialize variables
    best = None
    best_cls = None
    best_cost = sys.maxint
    nbr_cls = None

    for i in xrange(numlocal):
        print '%d%% done...' % int(i * (1. / numlocal) * 100)
        # Select random points
        cur = np.random.permutation(range(n))[:k]
        cur_cls = assign_to_closest(nodes, cur, dist)
        cur_cost = total_dist(nodes, cur_cls, dist)

        j = 0
        while j < maxneighbor:
            # Select random neighbor
            while True:
                rand = np.random.randint(0,n)
                if not rand in cur:
                    break
            rand_idx = np.random.randint(0,k)
            nbr = cur.copy()
            nbr[rand_idx] = rand

            # Check if with lower cost
            nbr_cls = assign_to_closest(nodes, nbr, dist, nbr_cls)
            nbr_cost = total_dist(nodes, nbr_cls, dist)
#            print 'best:', best_cost, 'cur:', cur_cost, 'nbr:', nbr_cost

            if nbr_cost < cur_cost:
                cur = nbr.copy()
                cur_cls = nbr_cls.copy()
                cur_cost = nbr_cost
                j = 0
            else:
                j += 1
        # Check if better than our current best
        if cur_cost < best_cost:
            best = cur.copy()
            best_cls = cur_cls.copy()
            best_cost = cur_cost

    print '100% done...'

    return best_cls, best, best_cost

def tri_ineq(nodes, dist, n, k, mds, mds_d_mat, cls=None, cls_dist=None, last_mds=None, last_cls=None, last_cls_dist=None, mds_tracking=None, cls_tracking=None, swap_idx=None, last_mds_d_mat=None):
    # Initialize stuff, if needed
    if cls == None:
        cls = np.empty(n)
    if cls_dist == None:
        cls_dist = np.empty(n)
    if cls_tracking == None:
        cls_tracking = np.empty(n)
    if mds_tracking == None:
        mds_tracking = np.empty(k)
    cls_dist[:] = sys.maxint
    cls_tracking[:] = 0

    # Cluster using previous medoid index
    if last_cls != None and last_mds != None and last_cls_dist != None and last_mds_d_mat != None and swap_idx != None:
        for node in xrange(n):
            if not node in last_mds:
                if last_cls[node] != last_mds[swap_idx]:
                    d = last_cls_dist[node]
                    idx = (mds==last_cls[node]).argmax()
                    if mds_d_mat[swap_idx,idx] >= 2*d:
                        cls[node] = last_cls[node]
                        cls_dist[node] = d
                        cls_tracking[node] = 1

    # Cluster using TIE
    for node in xrange(n):
        mds_tracking[:] = 0
        if cls_tracking[node] == 0:
            if not node in mds:
                for med1 in xrange(k):
                    if mds_tracking[med1] != 1:
                        d = dist[nodes[node]][nodes[mds[med1]]]
                        if d < cls_dist[node]:
                            cls[node] = mds[med1]
                            cls_dist[node] = d
                        for med2 in range(k)[med1+1:]:
                            if mds_d_mat[med1,med2] >= 2*d:
                                mds_tracking[med2] = 1
            else:
                cls[node] = node
                cls_dist[node] = 0.

    return cls, cls_dist
    
def testing(nodes, mds, dist):
    n = len(nodes)
    k = len(mds)
    cls = np.empty(n)
    cls_dist = np.empty(n)
    cls_dist[:] = sys.maxint
    for i in xrange(n):
        if i in mds:
            cls[i] = i
            cls_dist[i] = 0
        else:
            for j in xrange(k):
                d = dist[nodes[i]][nodes[mds[j]]]
                if d < cls_dist[i]:
                    cls_dist[i] = d
                    cls[i] = mds[j]

    return cls, cls_dist

def clarans_itp(nodes, k, numlocal, minmaxneighbor=250, p=0.0125, dist=dist_euc):
    if p < 0 or p > 1:
        raise ValueError('Argument p has to be in range [0,1].')

    # Calculate the maxneighbor attribute
    n = len(nodes)
    if k * (n-k) <= minmaxneighbor:
        maxneighbor = k * (n-k)
    else:
        maxneighbor = minmaxneighbor + int(p * (k * (n - k)))

    print 'CLARANS-ITP[numlocal:%d,maxneighbor:%d] computing...' % (numlocal, maxneighbor)

    # Initialize variables
    best = None
    best_cls = None
    best_cost = sys.maxint
    cur_cls = None
    cur_cls_dist = None
    nbr_cls = None
    nbr_cls_dist = None
    mds_tracking = None
    cls_tracking = None
    cur_mds_d_mat = np.asmatrix(np.zeros((k,k)))

    for i in xrange(numlocal):
        print '%d%% done...' % int(i * (1. / numlocal) * 100)
        # Select random points
        cur = np.random.permutation(range(n))[:k]
        # Calculate distances between medoids
        for x in xrange(k-1):
            for y in range(k)[x+1:]:
                if x != y:
                	d = dist[nodes[cur[x]]][nodes[cur[y]]]
                	cur_mds_d_mat[x,y] = d
                	cur_mds_d_mat[y,x] = d # TODO maybe just not do this? Use sparse matrix?

        cur_cls, cur_cls_dist = tri_ineq(nodes, dist, n, k, cur, cur_mds_d_mat, cur_cls, cur_cls_dist, mds_tracking=mds_tracking)
        cur_cost = np.sum(cur_cls_dist)

        j = 0
        while j < maxneighbor:
            # Select random neighbor
            while True:
                rand = np.random.randint(0,n)
                if not rand in cur:
                    break
            rand_idx = np.random.randint(0,k)
            nbr = cur.copy()
            nbr[rand_idx] = rand

            # Update distances between medoids
            nbr_mds_d_mat = cur_mds_d_mat.copy()
            for x in xrange(k):
                if x != rand_idx:
                    d = dist[nodes[rand]][nodes[nbr[x]]]
                    nbr_mds_d_mat[x,rand_idx] = d
                    nbr_mds_d_mat[rand_idx,x] = d

            nbr_cls, nbr_cls_dist = tri_ineq(nodes, dist, n, k, nbr, nbr_mds_d_mat, nbr_cls, nbr_cls_dist, cur, cur_cls, cur_cls_dist, mds_tracking, cls_tracking, rand_idx, cur_mds_d_mat)
            nbr_cost = np.sum(nbr_cls_dist)
#            print 'best_cost =', best_cost, 'cur_cost =', cur_cost, 'nbr_cost =', nbr_cost

            if nbr_cost < cur_cost:
                cur = nbr.copy()
                cur_cls = nbr_cls.copy()
                cur_cls_dist = nbr_cls_dist.copy()
                cur_cost = nbr_cost
                cur_mds_d_mat = nbr_mds_d_mat # TODO is this ok?
                j = 0
            else:
                j += 1

        # Check if better than our current best
        if cur_cost < best_cost:
            best = cur.copy()
            best_cls = cur_cls.copy()
            best_cost = cur_cost

    print '100% done...'

    return best_cls, best, best_cost