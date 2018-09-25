# Now functions related to threshold computation.

import numpy as np
from tqdm import tqdm_notebook as tqdm
from numpy.linalg  import norm
from scipy.sparse import csr_matrix

class ThresholdError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# There are two kinds of algorithm: psr1 and psr2 (psr=power spectral radius). They're both weighted and unweighted, so psr1uw, psr1w, psr2uw, psr2w
# psr1 is more crude: it checks convergence on the value of the spectral radius itself.
# psr2 is checks convergence on the principal eigenvector

# PSR1

# unweighted
def psr1uw(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose=False):
    # parameters
    # valumax = kwargs['valumax'] # 1000
    # tolerance = kwargs['tolerance'] # 1e-5
    # store = kwargs['store'] # 10

    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize
    leval = np.empty(shape=(store,), dtype=dtype)
    ceval = 0
    v0 = 0.9 * np.random.random(N) + 0.1
    v0 = np.array(v0, dtype=dtype)
    v = v0.copy()
    vold = v.copy()
    interrupt = False  # When convergence is reached, it becomes True.

    itmax = T * valumax
    for k in range(itmax):
        # Perform iteration:
        v = ladda * lA[k % T].dot(v) + (1. - mu) * v

        # Whether period is completed:
        if k % T == T - 1:
            # autoval = np.dot(vold,v)
            leval[ceval % store] = np.dot(vold, v) ** rootT
            ceval += 1
            # leval.append(autoval**rootT)

            # Check convergence
            if ceval >= store:
                fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
            else:
                fluct = 1. + tolerance
            if fluct < tolerance:
                interrupt = True
                break
            mnorm = np.linalg.norm(v)
            v = v / mnorm
            vold = v.copy()

    # If never interrupted, check now convergence.
    if not interrupt:
        fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
        if fluct >= tolerance:
            raise ThresholdError, 'Power method did not converge.'

    return leval[-1] - sr_target


# weighted
def psr1w(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose = False):
    # parameters
    # valumax = kwargs['valumax'] # 1000
    # tolerance = kwargs['tolerance'] # 1e-5
    # store = kwargs['store'] # 10

    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])
    loglad = np.log(1. - ladda)

    # Initialize
    leval = np.empty(shape=(store,), dtype=dtype)
    ceval = 0
    v0 = 0.9 * np.random.random(N) + 0.1
    v0 = np.array(v0, dtype=dtype)
    v = v0.copy()
    vold = v.copy()
    interrupt = False  # When convergence is reached, it becomes True.

    itmax = T * valumax
    for k in range(itmax):

        # Perform iteration. Meaning of function expm1: -(loglad*lA[k%T]).expm1() = 1-(1-ladda)^Aij
        v = -np.expm1(loglad * lA[k % T]).dot(v) + (1. - mu) * v

        # Whether period is completed
        if k % T == T - 1:
            leval[ceval % store] = np.dot(vold, v) ** rootT
            ceval += 1

            # Check convergence
            if ceval >= store:
                fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
            else:
                fluct = 1. + tolerance
            if fluct < tolerance:
                interrupt = True
                break
            mnorm = np.linalg.norm(v)
            v = v / mnorm
            vold = v.copy()

    # If never interrupted, check now convergence.
    if not interrupt:
        fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
        if fluct >= tolerance:
            raise ThresholdError, 'Power method did not converge.'
    return leval[-1] - sr_target


# PSR2

# unweighted
def psr2uw(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose = False):
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period

    # parameters
    # valumax = kwargs['valumax'] # 20000
    # tolerance = kwargs['tolerance'] # 1e-6
    # store = kwargs['store'] # 10
    # sr_target is usually=1. It's the target value for the spectral radius
    # Unless I'm discarding some empty timestep, in that case I want it to be (1-mu)^{-tau} where tau is how many I discard

    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize eigenvector register
    MV = np.empty(shape=(N, store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:, 0] = np.array([1. / np.sqrt(N)] * N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:, c].copy()

    for k in range(T * valumax):
        # Perform iteration:
        v = ladda * lA[k % T].dot(v) + (1. - mu) * v

        # Whether period is completed:
        if k % T == T - 1:

            # spectral radius
            sr = np.dot(MV[:, c % store], v)

            # normalize
            v = v / np.linalg.norm(v)

            # Compute tolerance, and return if reached:
            delta = np.sum(np.abs(MV[:, c % store] - v))
            if delta < tolerance:
                # return spectral radius^(1/T) - 1, as usual.
                return sr ** rootT - sr_target  # , v/np.linalg.norm(v)

            # increment index, and update storage
            c += 1
            MV[:, c % store] = v.copy()

    # if it goes out of the loop without returning the sr
    raise ThresholdError, 'Power method did not converge.'


# weighted
def psr2w(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose = False):
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period

    # parameters
    # valumax = kwargs['valumax'] # 20000
    # tolerance = kwargs['tolerance'] # 1e-6
    # store = kwargs['store'] # 10

    loglad = np.log(1. - ladda)
    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize eigenvector register
    MV = np.empty(shape=(N, store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:, 0] = np.array([1. / np.sqrt(N)] * N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:, c].copy()

    for k in range(T * valumax):
        # Perform iteration:
        v = -np.expm1(loglad * lA[k % T]).dot(v) + (1. - mu) * v

        # Whether period is completed:
        if k % T == T - 1:

            # spectral radius
            sr = np.dot(MV[:, c % store], v)

            # normalize
            v = v / np.linalg.norm(v)

            # Compute tolerance, and return if reached:
            delta = np.sum(np.abs(MV[:, c % store] - v))
            if delta < tolerance:
                # return spectral radius^(1/T) - 1, as usual.
                return sr ** rootT - sr_target  # , v/np.linalg.norm(v)

            # increment index, and update storage
            c += 1
            MV[:, c % store] = v.copy()

    # if it goes out of the loop without returning the sr
    raise ThresholdError, 'Power method did not converge.'


# for the aggregated spectral radius
def psr2uw_agg(A, N, valumax, tolerance, store):
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period

    # same data type as the adjacency matrices
    dtype = type(A[0, 0])

    # Initialize eigenvector register
    MV = np.empty(shape=(N, store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:, 0] = np.array([1. / np.sqrt(N)] * N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:, 0].copy()

    while c < valumax:
        # Perform iteration:
        v = A.dot(v)

        # spectral radius
        sr = np.dot(MV[:, c % store], v)

        # normalize
        v = v / np.linalg.norm(v)

        # Compute tolerance, and return if reached:
        delta = np.sum(np.abs(MV[:, c % store] - v))
        if delta < tolerance:
            # return spectral radius^(1/T) - 1, as usual.
            return sr

        # increment index, and update storage
        c += 1
        MV[:, c % store] = v.copy()

    # if it goes out of the loop without returning the sr
    raise ThresholdError, 'Power method did not converge.'





















# ###################################################################################################
# 
#                         EXPERIMENTAL
#
# ###################################################################################################

# There are two ADDITIONAL kinds of algorithm: psr3 and psr4 (psr=power spectral radius). They're both UNWEIGHTED, so psr3uw, psr4uw
# Both algorithms expect a fully prepared infection propagator matrix, i.e. NO epidemic parameters are required.
# psr3 is similar to psr1: it checks convergence on the value of the spectral radius itself.
# psr4 is similar to psr2 and checks convergence on the principal eigenvector

# PSR3

# unweighted
def psr3uw(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose=False):
    # parameters
    # valumax = kwargs['valumax'] # 1000
    # tolerance = kwargs['tolerance'] # 1e-5
    # store = kwargs['store'] # 10
    # diag: list of numpy.ndarrays. Each has length N - same dimension as 'v' 

    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize
    leval = np.empty(shape=(store,), dtype=dtype)
    ceval = 0
    v0 = 0.9 * np.random.random(N) + 0.1
    v0 = np.array(v0, dtype=dtype)
    v = v0.copy()
    vold = v.copy()
    interrupt = False  # When convergence is reached, it becomes True.

    itmax = T * valumax
    for k in range(itmax):
        # Perform iteration:
        v = ladda * lA[k % T].dot(v) + (1. - mu) * (1. - ladda * diag[k % T]) * v

        # Whether period is completed:
        if k % T == T - 1:
            # autoval = np.dot(vold,v)
            leval[ceval % store] = np.dot(vold, v) ** rootT
            ceval += 1
            # leval.append(autoval**rootT)

            # Check convergence
            if ceval >= store:
                fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
            else:
                fluct = 1. + tolerance
            if fluct < tolerance:
                interrupt = True
                break
            mnorm = np.linalg.norm(v)
            v = v / mnorm
            vold = v.copy()

    # If never interrupted, check now convergence.
    if not interrupt:
        fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
        if fluct >= tolerance:
            raise ThresholdError, 'Power method did not converge.'

    return leval[-1] - sr_target


# PSR3

# weighted
def psr3w(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose=False):
    # parameters
    # valumax = kwargs['valumax'] # 1000
    # tolerance = kwargs['tolerance'] # 1e-5
    # store = kwargs['store'] # 10
    # diag: list of numpy.ndarrays. Each has length N - same dimension as 'v' 

    loglad = np.log(1. - ladda)
    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize
    leval = np.empty(shape=(store,), dtype=dtype)
    ceval = 0
    v0 = 0.9 * np.random.random(N) + 0.1
    v0 = np.array(v0, dtype=dtype)
    v = v0.copy()
    vold = v.copy()
    interrupt = False  # When convergence is reached, it becomes True.

    itmax = T * valumax
    for k in range(itmax):
        # Perform iteration:
        v = -np.expm1(loglad * lA[k % T]).dot(v) + (1. - mu) * (1. + np.expm1(loglad * diag[k % T]) ) * v

        # Whether period is completed:
        if k % T == T - 1:
            # autoval = np.dot(vold,v)
            leval[ceval % store] = np.dot(vold, v) ** rootT
            ceval += 1
            # leval.append(autoval**rootT)

            # Check convergence
            if ceval >= store:
                fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
            else:
                fluct = 1. + tolerance
            if fluct < tolerance:
                interrupt = True
                break
            mnorm = np.linalg.norm(v)
            v = v / mnorm
            vold = v.copy()

    # If never interrupted, check now convergence.
    if not interrupt:
        fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
        if fluct >= tolerance:
            raise ThresholdError, 'Power method did not converge.'

    return leval[-1] - sr_target



# PSR4

# unweighted

def psr4uw(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose=False):
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period

    # parameters
    # valumax = kwargs['valumax'] # 20000
    # tolerance = kwargs['tolerance'] # 1e-6
    # store = kwargs['store'] # 10
    # sr_target is usually=1. It's the target value for the spectral radius
    # Unless I'm discarding some empty timestep, in that case I want it to be (1-mu)^{-tau} where tau is how many I discard

    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize eigenvector register
    MV = np.empty(shape=(N, store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:, 0] = np.array([1. / np.sqrt(N)] * N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:, c].copy()

    for k in range(T * valumax):
        # Perform iteration:
        v = ladda * lA[k % T].dot(v) + (1. - mu) * (1. - ladda * diag[k % T]) * v

        # Whether period is completed:
        if k % T == T - 1:

            # spectral radius
            sr = np.dot(MV[:, c % store], v)

            # normalize
            v = v / np.linalg.norm(v)

            # Compute tolerance, and return if reached:
            delta = np.sum(np.abs(MV[:, c % store] - v))
            if delta < tolerance:
                # return spectral radius^(1/T) - 1, as usual.
                return sr ** rootT - sr_target  # , v/np.linalg.norm(v)

            # increment index, and update storage
            c += 1
            MV[:, c % store] = v.copy()

    # if it goes out of the loop without returning the sr
    raise ThresholdError, 'Power method did not converge.'

'''
def psr4uw(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose=False):
    """
    lip: indptr
    li: indices
    ld: dataMVMV
    lp: places
    diag: diagonal
    """
    lip = np.concatenate( [A.indptr for A in lA] )
    li = np.concatenate( [A.indices for A in lA] )
    ld = np.concatenate( [A.data for A in lA] )

    c_abs = np.abs #CHANGE!!!


    rootT = T
    rootT = 1./rootT

    # counters and indices and util
    #cdef int i,k,a,tau
    #cdef mydouble sr,norm,delta

    # eigenvector register. c-th vector is MV[c*N:(c+1)*N]. Vector to use
    #cdef mydouble *MV
    #cdef mydouble *v
    #cdef mydouble *v2
    #MV = <mydouble *>malloc(N*store*cython.sizeof(mydouble))
    #v  = <mydouble *>malloc(N*cython.sizeof(mydouble))
    #v2 = <mydouble *>malloc(N*cython.sizeof(mydouble))
    MV = np.zeros(N*store, dtype=np.float128)
    v  = np.zeros(N, dtype=np.float128)
    v2 = np.zeros(N, dtype=np.float128)
    
    c = 0
    #cdef int idx
    #cdef mydouble diag_element

    i = 0
    MV[0] = 1. #REMOVE!!!
    MV[1] = 0. #REMOVE!!!
    while i<N:
        #MV[i] = 1./(N ** 0.5)  #UNCOMMENT!!!
        v[i] = MV[i]
        i += 1

    flag = 0
    #cdef mydouble ecco

    #with nogil:
    k = 0
    while flag != 1 and k < T*valumax:

        # time step
        tau = k%T
        
        i = 0
        while i<N:
            # diagonal
            #idx = i + tau * N
            #diag_element = diag[idx]

            v2[i] = (1. - mu) * (1.-ladda * diag[tau,i] ) * v[i]
            
            #if diag_element > 0:
            #    v2[i] = (1.-ladda * diag[idx] ) * (1. - mu) * v[i]
            #elif diag_element == 0:
            #    v2[i] = (1. - mu) * v[i]
            #else:
            #    print "wrong diagonal value:", diag_element
            #    raise ValueError
            #assert v2[i] > 0, "Value error: v2[i] = {}".format(v2[i])

            #idx = i+tau*(N+1)
            # multiply
            a = lip[i+tau*(N+1)]
            while a<lip[i+1+tau*(N+1)]:
                v2[i] += ladda*ld[a]*v[li[a]]
                a += 1
            i += 1
            

        # copy back to v
        i = 0
        while i<N:
            v[i] = v2[i]
            i += 1

        # if period is complete
        if tau == T-1:

            # compute sr approximation, and norm of v
            sr = 0.
            norm = 0.
            i = 0
            while i<N:
                sr += v[i]*MV[c*N+i]
                norm += v[i]*v[i]
                i += 1
            norm = norm ** 0.5

            # normalize v, and compute L1-norm
            i = 0
            delta = 0.
            while i<N:
                v[i] = v[i]/norm
                delta += c_abs(MV[c*N+i]-v[i])
                i += 1

            if delta<tolerance:
                flag = 1
                ecco = sr**rootT - sr_target

            if flag!=1:
                # put in store
                c = (c+1)%store
                i = 0
                while i<N:
                    MV[c*N+i] = v[i]
                    i += 1

        k += 1

    #free(MV)
    #free(v)
    #free(v2)


    #if flag != 1:
    #    raise ValueError

    return ecco
'''

# PSR4

# unweighted
def psr4w(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target, diag, verbose=False):
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period

    # parameters
    # valumax = kwargs['valumax'] # 20000
    # tolerance = kwargs['tolerance'] # 1e-6
    # store = kwargs['store'] # 10
    # sr_target is usually=1. It's the target value for the spectral radius
    # Unless I'm discarding some empty timestep, in that case I want it to be (1-mu)^{-tau} where tau is how many I discard

    loglad = np.log(1. - ladda)
    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize eigenvector register
    MV = np.empty(shape=(N, store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:, 0] = np.array([1. / np.sqrt(N)] * N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:, c].copy()

    for k in range(T * valumax):
        # Perform iteration:
        v = -np.expm1(loglad * lA[k % T]).dot(v) + (1. - mu) * (1. + np.expm1(loglad * diag[k % T]) ) * v

        # Whether period is completed:
        if k % T == T - 1:
            
            # spectral radius
            sr = np.dot(MV[:, c % store], v)

            # normalize
            v = v / np.linalg.norm(v)

            # Compute tolerance, and return if reached:
            delta = np.sum(np.abs(MV[:, c % store] - v))
            if delta < tolerance:
                # return spectral radius^(1/T) - 1, as usual.
                return sr ** rootT - sr_target  # , v/np.linalg.norm(v)

            # increment index, and update storage
            c += 1
            MV[:, c % store] = v.copy()

    # if it goes out of the loop without returning the sr
    raise ThresholdError, 'Power method did not converge.'


if __name__ == "__main__":
    pass