cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport log, exp, pow
from libc.math cimport fabs as c_abs
import warnings

import numpy
cimport numpy

ctypedef long double mydouble
ctypedef long myint

#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.cdivision(True)
cpdef mydouble psr2uw(mydouble ladda, mydouble [:] mu, int [:] lip, int [:] li, mydouble [:] ld, int [:] lp, myint N, myint T, myint valumax, mydouble tolerance, myint store, mydouble sr_target):
    """
    lip: indptr
    li: indices
    ld: data
    lp: places
    """

    cdef mydouble rootT = T
    rootT = 1./rootT

    # counters and indices and util
    cdef int i,k,a,tau
    cdef mydouble sr,norm,delta

    # eigenvector register. c-th vector is MV[c*N:(c+1)*N]. Vector to use
    cdef mydouble *MV
    cdef mydouble *v
    cdef mydouble *v2
    MV = <mydouble *>malloc(N*store*cython.sizeof(mydouble))
    v  = <mydouble *>malloc(N*cython.sizeof(mydouble))
    v2 = <mydouble *>malloc(N*cython.sizeof(mydouble))
    cdef int c = 0
    i = 0
    while i<N:
        MV[i] = 1./(N ** 0.5)
        v[i] = MV[i]
        i += 1

    cdef int flag = 0
    cdef int flag_negativ_element_warning = 0
    cdef mydouble ecco

    with nogil:
        k = 0
        while flag != 1 and k < T*valumax:

            # time step
            tau = k%T

            i = 0
            while i<N:
                # diagonal
                v2[i] = (1.-mu[i])*v[i]

                # multiply
                a = lip[i+tau*(N+1)] + lp[tau] #CHANGED
                while a<lip[i+1+tau*(N+1)] + lp[tau]: #CHANGED
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
                    i = 0
                    while i<N:
                        if v[i] < 0:
                            flag_negativ_element_warning = 1
                        i += 1
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

        free(MV)
        free(v)
        free(v2)


    if flag != 1:
        raise ValueError
    if flag_negativ_element_warning:
        warnings.warn("Value error: vector element was negativ")

    return ecco

# ###################################################################################################
# 
#                         Individual-based model WEIGHTED
#
# ###################################################################################################


# @cython.boundscheck(False)
cpdef mydouble psr2w(mydouble ladda, mydouble [:] mu, int [:] lip, int [:] li, mydouble [:] ld, int [:] lp, myint N, myint T, myint valumax, mydouble tolerance, myint store, mydouble sr_target):
    """
    lip: indptr
    li: indices
    ld: data
    lp: places
    """

    cdef mydouble laddam1 = 1. - ladda
    cdef mydouble rootT = T
    rootT = 1./rootT

    # counters and indices and util
    cdef int i,k,a,tau
    cdef mydouble sr,norm,delta

    # eigenvector register. c-th vector is MV[c*N:(c+1)*N]. Vector to use
    cdef mydouble *MV
    cdef mydouble *v
    cdef mydouble *v2
    MV = <mydouble *>malloc(N*store*cython.sizeof(mydouble))
    v  = <mydouble *>malloc(N*cython.sizeof(mydouble))
    v2 = <mydouble *>malloc(N*cython.sizeof(mydouble))
    cdef int c = 0
    i = 0
    while i<N:
        MV[i] = 1./(N ** 0.5)
        v[i] = MV[i]
        i += 1

    cdef int flag = 0
    cdef int flag_negativ_element_warning = 0
    cdef mydouble ecco

    with nogil:
        k = 0
        while flag != 1 and k < T*valumax:

            # time step
            tau = k%T

            i = 0
            while i<N:
                # diagonal
                v2[i] = (1.-mu[i])*v[i]

                # multiply
                a = lip[i+tau*(N+1)] + lp[tau] #CHANGED
                while a<lip[i+1+tau*(N+1)] + lp[tau]: #CHANGED
                    v2[i] += ( 1. - pow(laddam1, ld[a]) ) * v[li[a]]
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
                    #if v[i] < 0:
                    #    flag_negativ_element_warning = 1
                    #if v[i] > 1:
                    #    flag_above_one_warning = 1
                    v[i] = v[i]/norm
                    delta += c_abs(MV[c*N+i]-v[i])
                    i += 1

                if delta<tolerance:
                    flag = 1
                    ecco = sr**rootT - sr_target
                    i = 0
                    while i<N:
                        if v[i] < 0:
                            flag_negativ_element_warning = 1
                        i += 1

                if flag!=1:
                    # put in store
                    c = (c+1)%store
                    i = 0
                    while i<N:
                        MV[c*N+i] = v[i]
                        i += 1

            k += 1

        free(MV)
        free(v)
        free(v2)

    if flag != 1:
        raise ValueError
    if flag_negativ_element_warning:
        warnings.warn("Value error: vector element was negativ")

    return ecco

# ###################################################################################################
# 
#                         Contact-based model UNWEIGHTED
#
# ###################################################################################################

# @cython.boundscheck(False)
cpdef mydouble psr4uw(mydouble ladda, mydouble [:] mu, int [:] lip, int [:] li, mydouble [:] ld, int [:] lp, myint N, myint T, myint valumax, mydouble tolerance, myint store, mydouble sr_target,  mydouble [:,:] diag):
    """
    lip: indptr
    li: indices
    ld: dataMVMV
    lp: places
    diag: diagonal
    """

    cdef mydouble rootT = T
    rootT = 1./rootT

    # counters and indices and util
    cdef int i,k,a,tau
    cdef mydouble sr,norm,delta

    # eigenvector register. c-th vector is MV[c*N:(c+1)*N]. Vector to use
    cdef mydouble *MV
    cdef mydouble *v
    cdef mydouble *v2
    MV = <mydouble *>malloc(N*store*cython.sizeof(mydouble))
    v  = <mydouble *>malloc(N*cython.sizeof(mydouble))
    v2 = <mydouble *>malloc(N*cython.sizeof(mydouble))
    cdef int c = 0
    cdef int idx
    cdef mydouble diag_element

    i = 0
    while i<N:
        MV[i] = 1./(N ** 0.5)
        v[i] = MV[i]
        i += 1

    cdef int flag = 0
    cdef int flag_above_one_warning = 0
    cdef int flag_negativ_element_warning = 0
    cdef mydouble ecco

    with nogil:
        k = 0
        while flag != 1 and k < T*valumax:

            # time step
            tau = k%T

            
            #v2 = np.multiply( (1. - mu) * (1. - ladda * diag[k % T : k % T + N]), v)

            i = 0
            while i<N:
                # diagonal
                #idx = i+tau*(N)
                #diag_element = diag[idx]

                v2[i] = (1. - mu[i]) * (1.-ladda * diag[tau,i] ) * v[i]
                
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
                a = lip[i+tau*(N+1)] + lp[tau] #CHANGED
                while a<lip[i+1+tau*(N+1)] + lp[tau]: #CHANGED
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
                    #if v[i] < 0:
                    #    flag_negativ_element_warning = 1
                    #if v[i] > 1:
                    #    flag_above_one_warning = 1
                    v[i] = v[i]/norm
                    delta += c_abs(MV[c*N+i]-v[i])
                    i += 1

                if delta<tolerance:
                    flag = 1
                    ecco = sr**rootT - sr_target
                    i = 0
                    while i<N:
                        if v[i] < 0:
                            flag_negativ_element_warning = 1
                        i += 1

                if flag!=1:
                    # put in store
                    c = (c+1)%store
                    i = 0
                    while i<N:
                        MV[c*N+i] = v[i]
                        i += 1

            k += 1

    free(MV)
    free(v)
    free(v2)

    if flag != 1:
        raise ValueError
    if flag_negativ_element_warning:
        warnings.warn("Value error: vector element was negativ")

    return ecco


# ###################################################################################################
# 
#                         Contact-based model WEIGHTED
#
# ###################################################################################################

# @cython.boundscheck(False)
cpdef mydouble psr4w(mydouble ladda, mydouble [:] mu, int [:] lip, int [:] li, mydouble [:] ld, int [:] lp, myint N, myint T, myint valumax, mydouble tolerance, myint store, mydouble sr_target,  mydouble [:,:] diag):
    """
    lip: indptr
    li: indices
    ld: dataMVMV
    lp: places
    diag: diagonal
    """

    cdef mydouble laddam1 = 1. - ladda
    cdef mydouble rootT = T
    rootT = 1./rootT

    # counters and indices and util
    cdef int i,k,a,tau
    cdef mydouble sr,norm,delta

    # eigenvector register. c-th vector is MV[c*N:(c+1)*N]. Vector to use
    cdef mydouble *MV
    cdef mydouble *v
    cdef mydouble *v2
    MV = <mydouble *>malloc(N*store*cython.sizeof(mydouble))
    v  = <mydouble *>malloc(N*cython.sizeof(mydouble))
    v2 = <mydouble *>malloc(N*cython.sizeof(mydouble))
    cdef int c = 0
    cdef int idx
    cdef mydouble diag_element
    
    i = 0
    while i<N:
        MV[i] = 1./(N ** 0.5)
        v[i] = MV[i]
        i += 1

    cdef int flag = 0
    cdef int flag_negativ_element_warning = 0
    cdef mydouble ecco

    with nogil:
        k = 0
        while flag != 1 and k < T*valumax:

            # time step
            tau = k%T

            i = 0
            while i<N:
                diag_element = diag[tau,i]
                # diagonal
                if diag_element > 0:
                    v2[i] = pow(laddam1, diag_element) * (1. - mu[i]) * v[i]
                elif diag_element == 0:
                    v2[i] = (1. - mu[i]) * v[i]
                #v2[i] = pow(laddam1, diag[i+tau*N]) * (1. - mu) * v[i]
                # print i+tau*(N+1)

                # multiply
                a = lip[i+tau*(N+1)] + lp[tau] #CHANGED
                while a<lip[i+1+tau*(N+1)] + lp[tau]: #CHANGED
                    v2[i] += ( 1. - pow(laddam1, ld[a]) ) * v[li[a]]
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
                    #if v[i] < 0:
                    #    flag_negativ_element_warning = 1
                    #if v[i] > 1:
                    #    flag_above_one_warning = 1
                    v[i] = v[i]/norm
                    delta += c_abs(MV[c*N+i]-v[i])
                    i += 1

                if delta<tolerance:
                    flag = 1
                    ecco = sr**rootT - sr_target
                    i = 0
                    while i<N:
                        if v[i] < 0:
                            flag_negativ_element_warning = 1
                        i += 1

                if flag!=1:
                    # put in store
                    c = (c+1)%store
                    i = 0
                    while i<N:
                        MV[c*N+i] = v[i]
                        i += 1

            k += 1

    free(MV)
    free(v)
    free(v2)
    if flag != 1:
        raise ValueError
    if flag_negativ_element_warning:
        warnings.warn("Value error: vector element was negativ")

    return ecco