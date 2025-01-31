from cvxopt import matrix, spdiag, mul, div, sqrt
from cvxopt import blas, lapack, solvers 
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import math

def l1regls(A, y):
    """
    
    Returns the solution of l1-norm regularized least-squares problem
  
        minimize || A*x - y ||_2^2  + || x ||_1.

    """

    m, n = A.size
    q = matrix(1.0, (2*n,1))
    q[:n] = -2.0 * A.T * y

    def P(u, v, alpha = 1.0, beta = 0.0 ):
        """
            v := alpha * 2.0 * [ A'*A, 0; 0, 0 ] * u + beta * v 
        """
        v *= beta
        v[:n] += alpha * 2.0 * A.T * (A * u[:n])


    def G(u, v, alpha=1.0, beta=0.0, trans='N'):
        """
            v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
        """

        v *= beta
        v[:n] += alpha*(u[:n] - u[n:])
        v[n:] += alpha*(-u[:n] - u[n:])

    h = matrix(0.0, (2*n,1))


    # Customized solver for the KKT system 
    #
    #     [  2.0*A'*A  0    I      -I     ] [x[:n] ]     [bx[:n] ]
    #     [  0         0   -I      -I     ] [x[n:] ]  =  [bx[n:] ].
    #     [  I        -I   -D1^-1   0     ] [zl[:n]]     [bzl[:n]]
    #     [ -I        -I    0      -D2^-1 ] [zl[n:]]     [bzl[n:]]
    #
    # where D1 = W['di'][:n]**2, D2 = W['di'][n:]**2.
    #    
    # We first eliminate zl and x[n:]:
    #
    #     ( 2*A'*A + 4*D1*D2*(D1+D2)^-1 ) * x[:n] = 
    #         bx[:n] - (D2-D1)*(D1+D2)^-1 * bx[n:] + 
    #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] - 
    #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:]           
    #
    #     x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] ) 
    #         - (D2-D1)*(D1+D2)^-1 * x[:n]         
    #
    #     zl[:n] = D1 * ( x[:n] - x[n:] - bzl[:n] )
    #     zl[n:] = D2 * (-x[:n] - x[n:] - bzl[n:] ).
    #
    # The first equation has the form
    #
    #     (A'*A + D)*x[:n]  =  rhs
    #
    # and is equivalent to
    #
    #     [ D    A' ] [ x:n] ]  = [ rhs ]
    #     [ A   -I  ] [ v    ]    [ 0   ].
    #
    # It can be solved as 
    #
    #     ( A*D^-1*A' + I ) * v = A * D^-1 * rhs
    #     x[:n] = D^-1 * ( rhs - A'*v ).

    S = matrix(0.0, (m,m))
    Asc = matrix(0.0, (m,n))
    v = matrix(0.0, (m,1))

    def Fkkt(W):

        # Factor 
        #
        #     S = A*D^-1*A' + I 
        #
        # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**-2, D2 = d[n:]**-2.

        d1, d2 = W['di'][:n]**2, W['di'][n:]**2

        # ds is square root of diagonal of D
        ds = math.sqrt(2.0) * div( mul( W['di'][:n], W['di'][n:]), 
            sqrt(d1+d2) )
        d3 =  div(d2 - d1, d1 + d2)
     
        # Asc = A*diag(d)^-1/2
        Asc = A * spdiag(ds**-1)

        # S = I + A * D^-1 * A'
        blas.syrk(Asc, S)
        S[::m+1] += 1.0 
        lapack.potrf(S)

        def g(x, y, z):

            x[:n] = 0.5 * ( x[:n] - mul(d3, x[n:]) + 
                mul(d1, z[:n] + mul(d3, z[:n])) - mul(d2, z[n:] - 
                mul(d3, z[n:])) )
            x[:n] = div( x[:n], ds) 

            # Solve
            #
            #     S * v = 0.5 * A * D^-1 * ( bx[:n] - 
            #         (D2-D1)*(D1+D2)^-1 * bx[n:] + 
            #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] - 
            #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:] )
                
            blas.gemv(Asc, x, v)
            lapack.potrs(S, v)
            
            # x[:n] = D^-1 * ( rhs - A'*v ).
            blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
            x[:n] = div(x[:n], ds)

            # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] ) 
            #         - (D2-D1)*(D1+D2)^-1 * x[:n]         
            x[n:] = div( x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1+d2 )\
                - mul( d3, x[:n] )
                
            # zl[:n] = D1^1/2 * (  x[:n] - x[n:] - bzl[:n] )
            # zl[n:] = D2^1/2 * ( -x[:n] - x[n:] - bzl[n:] ).
            z[:n] = mul( W['di'][:n],  x[:n] - x[n:] - z[:n] ) 
            z[n:] = mul( W['di'][n:], -x[:n] - x[n:] - z[n:] ) 

        return g

    return solvers.coneqp(P, q, G, h, kktsolver = Fkkt)['x'][:n]

class AR:
    def __init__(self):
        pass

    def fit(self, y, p = 2):
        n = len(y)
        X = np.column_stack([y[p - k - 1 : n - k - 1] for k in range(p)])
        yt = y[p:]
        
        # self.c = np.linalg.inv((X.T @ X)) @ X.T @ yt
        self.c = sp.linalg.lstsq(X, yt, cond = None)[0]
    
    def lasso(self, y, p = 2):
        n = len(y)
        X = np.column_stack([y[p - k - 1 : n - k - 1] for k in range(p)])
        yt = y[p:]

        solution = l1regls(matrix(X), matrix(yt))
        self.c = np.zeros(solution.size)
        self.c[:solution.size[0], :solution.size[1]] = solution[:, :]
        self.c = self.c.flatten()

    def predict(self, y, steps = 1):
        p = len(self.c)
        yx = np.concatenate([y, np.zeros(steps)])

        for t in range(steps):
            yx[len(y) + t] = np.dot(self.c, yx[len(y) + t - p:len(y) + t][::-1])

        return yx[-steps:]

if __name__ == '__main__':
    np.random.seed(1)
    N = 1000
    time = np.arange(N)

    trend = 0.0001 * time**2 - 0.005 * time + 5
    amplitude = 0.25 + 0.001 * time  
    frequency = 0.00001 * time 
    sezon = amplitude * np.sin(2 * np.pi * frequency * time)
    variatii = np.random.normal(0, 1, N)
    y = trend + sezon + variatii
    
    ar = AR()
    ar.fit(y, 5)

    preds = [y[-1]]
    preds.extend(ar.predict(y, 150))
    plt.plot(y, label = '$y$')
    plt.plot(range(len(y) - 1, len(y) + len(preds) - 1), preds, label = '$\^{y}$')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./labs/12/02.pdf')
    plt.show()