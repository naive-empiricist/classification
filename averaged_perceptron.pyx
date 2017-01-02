import logging
import scipy
from scipy.sparse import csr_matrix
import random
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, log, fabs
from cpython cimport bool
cimport cython
from cython.view cimport array as cvarray

ctypedef np.float32_t dtype_t

# x += a*y, where x is dense and y is sparse.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef public void add_weighted(double[:] x, double[:] ydata , int[:] yindices, int ylen, double a):
    cdef unsigned int i
    
    for i in range(ylen):
        #print "i: %d" % i
        #print "ylen: %d" % ylen
        #print "ydata: %1.1e" % ydata[i]
        x[yindices[i]] += a*ydata[i]

# Dot product of a dense and a sparse vector
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef public double spdot(double[:] x, double[:] ydata , int[:] yindices, int ylen):
    cdef unsigned int i
    cdef double v = 0.0
    
    for i in range(ylen):
        v += ydata[i]*x[yindices[i]]
        
    return v


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def averaged_perceptron_multi(Xn, int [:] dn, props):
            
    cdef int [:] indices, indptr
    cdef double [:] data
    cdef int i
    
    if not isinstance(Xn, csr_matrix):
        raise Exception("Averaged Perceptron requires a CSR matrix")
        
    indices = Xn.indices.astype(np.int32)
    indptr = Xn.indptr.astype(np.int32)
    data = Xn.data.astype(np.float64)
                
    cdef int [:] d = dn
    
    cdef double [:] rdata
    cdef int [:] rind
    cdef double sgn, bestScore, score
    cdef int ry, rhat
    cdef int rlen
    cdef int processed = 0
    cdef int iteration = 0
    cdef double current = 0.0
    cdef int errors = 0
    
                
    logger = logging.getLogger("averaged_perceptron")

    cdef int npoints = Xn.shape[0] # points
    cdef int ndims = Xn.shape[1] # dims
    cdef int nlabels = len(set(d))

    cdef int maxIters = props.get("passes", 10)

    cdef double [:,:] w = np.zeros((nlabels, ndims))
    cdef double [:,:] wbar = np.zeros((nlabels, ndims))

    logger.info("Averaged Perceptron starting, npoints=%d, ndims=%d", npoints, ndims)  
    print("Averaged Perceptron starting, npoints=%d, ndims=%d", npoints, ndims) 
    
    while iteration < maxIters:
        #logger.info("starting iteration %d" % iteration)
        #print("starting iteration %d" % iteration)
        errors = 0
        for j in range(npoints):
            i = np.random.randint(0, npoints)
            
            #t = j+iteration*npoints+1
            #eta = stepSize/t # Step size
            
            rlen = indptr[i+1]-indptr[i]
            rind = indices[indptr[i]:indptr[i+1]]
            rdata = data[indptr[i]:indptr[i+1]]
            ry = d[i]
            
            # get argmax of dot products
            rhat = 0
            bestScore = spdot(w[0], rdata , rind, rlen)
            for i in range(1, nlabels):
                score = spdot(w[i], rdata , rind, rlen)
                if score > bestScore:
                    bestScore = score
                    rhat = i
            
            #print(np.asarray(w), np.asarray(rdata), np.asarray(rind), rlen, ry, rhat)
            if rhat != ry:
                errors += 1
                add_weighted(w[ry], rdata, rind, rlen, 1.0)
                add_weighted(wbar[ry], rdata, rind, rlen, current)
                add_weighted(w[rhat], rdata, rind, rlen, -1.0)
                add_weighted(wbar[rhat], rdata, rind, rlen, - current)
            current += 1.0
        
        #logger.info("AP -- errors: %d (%2.3f percent)" % (errors, 100.0*errors/(0.0+npoints))) 
        #logger.info("Epoch %d finished" % iteration)
        print("AP -- errors: %d (%2.3f percent)" % (errors, 100.0*errors/(0.0+npoints))) 
        print("Epoch %d finished" % iteration)
        iteration = iteration + 1
                
    logger.info("Averaged Perceptron finished")
    
    #wbar /= current
    cdef double[:,:] f = np.subtract(w, np.divide(wbar, current))

    return f

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def averaged_perceptron(Xn, double [:] dn, props):
            
    cdef int [:] indices, indptr
    cdef double [:] data
    cdef int i
    
    if not isinstance(Xn, csr_matrix):
        raise Exception("Averaged Perceptron requires a CSR matrix")
        
    indices = Xn.indices.astype(np.int32)
    indptr = Xn.indptr.astype(np.int32)
    data = Xn.data.astype(np.float64)
                
    cdef double [:] d = dn
    
    cdef double [:] rdata
    cdef int [:] rind
    cdef double sgn, rhat
    cdef int rlen
    cdef int processed = 0
    cdef int iteration = 0
    cdef double current = 0.0
    cdef int errors = 0
                
    logger = logging.getLogger("averaged_perceptron")

    cdef int npoints = Xn.shape[0] # points
    cdef int ndims = Xn.shape[1] # dims

    cdef int maxIters = props.get("passes", 10)

    cdef double [:] w = np.zeros(ndims)
    cdef double [:] wbar = np.zeros(ndims)

    logger.info("Averaged Perceptron starting, npoints=%d, ndims=%d", npoints, ndims)

    while iteration < maxIters:
        #logger.info("starting iteration %d" % iteration)
        #print("starting iteration %d" % iteration)
        errors = 0
        for j in range(npoints):
            i = np.random.randint(0, npoints)
            
            #t = j+iteration*npoints+1
            #eta = stepSize/t # Step size
            
            rlen = indptr[i+1]-indptr[i]
            rind = indices[indptr[i]:indptr[i+1]]
            rdata = data[indptr[i]:indptr[i+1]]
            ry = d[i]
            
            rhat = spdot(w, rdata , rind, rlen)
            #print("ry=%f -- rhat=%f" % (ry, rhat))
            #print(np.asarray(w), np.asarray(rdata), np.asarray(rind), rlen, ry, rhat)
            #sgn = rhat * ry >= 0 else -1.0
            if rhat * ry <= 0:
                errors += 1
                sgn = 1.0 if ry >= 0 else -1.0
                add_weighted(w, rdata, rind, rlen, sgn)
                add_weighted(wbar, rdata, rind, rlen, sgn * current)
            current += 1.0
        
        #logger.info("AP -- errors: %d (%2.3f percent)" % (errors, 100.0*errors/(0.0+npoints))) 
        #logger.info("Epoch %d finished" % iteration)
        print("AP -- errors: %d (%2.3f percent)" % (errors, 100.0*errors/(0.0+npoints))) 
        print("Epoch %d finished" % iteration)
        iteration = iteration + 1
                
    logger.info("Averaged Perceptron finished")
    
    #wbar /= current
    cdef double[:] f = np.subtract(w, np.divide(wbar, current))

    return f




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
def test_averaged_perceptron_multi(Xn, int [:] dn, double[:,:] model):
            
    cdef int [:] indices, indptr
    cdef double [:] data
    cdef int i
    
    if not isinstance(Xn, csr_matrix):
        raise Exception("Averaged Perceptron requires a CSR matrix")
        
    indices = Xn.indices.astype(np.int32)
    indptr = Xn.indptr.astype(np.int32)
    data = Xn.data.astype(np.float64)
                
    cdef int [:] d = dn
    
    cdef double [:] rdata
    cdef int [:] rind
    cdef double sgn, bestScore, score
    cdef int ry, rhat, lbl
    cdef int rlen
    
                
    logger = logging.getLogger("test_averaged_perceptron")

    cdef int npoints = Xn.shape[0] # points
    cdef int ndims = Xn.shape[1] # dims
    cdef int nlabels = len(set(d))

    #logger.info("Averaged Perceptron testing, npoints=%d", npoints)  
    print("Averaged Perceptron testing, npoints =", npoints, "ndims =", ndims, " labels =", nlabels)  
    
    cdef int errors = 0
    cdef int current = 0
    for i in range(npoints):
            
        rlen = indptr[i+1]-indptr[i]
        rind = indices[indptr[i]:indptr[i+1]]
        rdata = data[indptr[i]:indptr[i+1]]
        ry = d[i]
            
        # get argmax of dot products
        rhat = 0
        bestScore = spdot(model[0], rdata , rind, rlen)
        for lbl in range(1, nlabels):
            score = spdot(model[lbl], rdata , rind, rlen)
            if score > bestScore:
                bestScore = score
                rhat = lbl
        
        current += 1
        #print("current =", current)
        if rhat != ry:
            errors += 1
            #print("  errors =", errors)
        
    #logger.info("AP -- errors: %d (%2.3f percent)" % (errors, 100.0*errors/(0.0+npoints))) 
    #logger.info("Epoch %d finished" % iteration)
        
    print("AP -- test errors: %d (%2.3f percent)" % (errors, 100.0*errors/float(npoints)))           
    #logger.info("Averaged Perceptron Multi Test finished")
    print("Averaged Perceptron Multi Test finished")



def ap_test():
    #cdef int [:] 
    indices = [0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2]
    #cdef double [:] 
    data =    [1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 4.0, 
               1.0, 3.0, 1.0, 2.0, 1.0, 7.0, 1.0, 1.0]
    #cdef int [:] 
    indptr = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    xs = csr_matrix((data, indices, indptr))
    #cdef double [:] 
    b = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
    barr = np.array(b, dtype=np.float64)
    return averaged_perceptron(xs, barr, {"passes":20})

def ap_test2(filename):
    ### data, indices, indptr)
    indptr = []
    indices = []
    data = []
    yarr = []
    indptr.append(0)
    with open(filename, 'r') as fin:
        for line in fin:
            l = line.strip()
            if len(l) == 0 or l.startswith("#"):
                continue
            l = l.split()
            yarr.append( float(l[0]) )
            for p in l[1:]:
                fv = p.split(":")
                if len(fv) != 2: continue
                indices.append( int(fv[0]) )
                data.append( float(fv[1]) )
            indptr.append(len(data))
    xs = csr_matrix((data, indices, indptr))
    ys = np.array(yarr, dtype=np.float64)
    return averaged_perceptron(xs, ys, {"passes":30})
    
def ap_test_multi(trainingFilename, testFilename):
    xs, ys = openFilename(trainingFilename)
    f = averaged_perceptron_multi(xs, ys, {"passes":30})
    xtest, ytest = openFilename(testFilename, f.shape[1])
    test_averaged_perceptron_multi(xtest, ytest, f)
    

def openFilename(filename, maxDim=np.inf):
    ### data, indices, indptr)
    indptr = []
    indices = []
    data = []
    yarr = []
    indptr.append(0)
    with open(filename, 'r') as fin:
        for line in fin:
            l = line.strip()
            if len(l) == 0 or l.startswith("#"):
                continue
            l = l.split()
            yarr.append( int(l[0]) )
            for p in l[1:]:
                fv = p.split(":")
                if len(fv) != 2: continue
                f = int(fv[0])
                if f < maxDim:
                    indices.append( f )
                    data.append( float(fv[1]) )
            indptr.append(len(data))
    xs = csr_matrix((data, indices, indptr))
    ys = np.array(yarr, dtype=np.int32)
    return xs, ys
