import numpy as np
import scipy.spatial.distance

# listes [[x, y], [x, y], [x, y]...]
def precision_recall(dets, gts, tolerance=3):
    dists = scipy.spatial.distance.cdist(dets, gts)
    idx = np.argsort(dists.flatten())
    ys = (idx / gts.shape[0]).astype(int)
    xs = (idx % gts.shape[0])
   
    affected_np = -np.ones(dets.shape[0], dtype=int)
    used_np = -np.ones(gts.shape[0], dtype=int)
    for i in range(ys.shape[0]):
        y = ys[i]
        x = xs[i]
        if (dists[y, x] > tolerance):
            break
        if (used_np[x] >= 0):
            continue
        
        affected_np[y] = x
        used_np[x] = y
        
    tp = np.sum(affected_np >= 0)
    precision = tp / dets.shape[0]
    recall = tp / gts.shape[0]
    
    return precision, recall




#
#gts = np.load('gt.npz')['points']
#
#
#print (precision_recall(np.random.random((100,2)) * 500, gts))