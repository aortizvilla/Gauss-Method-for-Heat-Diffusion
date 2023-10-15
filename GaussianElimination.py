import numpy as np

def eliminacioGaussiana(A, b):
    '''Eliminació Gaussiana. Es modifiquen A i b'''
    n=len(A)
    if b.size != n:
        raise ValueError("Error: les dimensions de A i b no coincideixen", b.size, n)
    for k in range(n-1):
        for i in range(k+1, n):
            alpha = A[i,k]/A[k,k]
            #Falta posar zeros a la resta de la columna, si es vol
            #A[i,k] = alpha si es vol guardar L per LU
            A[i,k]=0
            for j in range(k+1, n):
                A[i,j] = A[i,j] - alpha*A[k,j]
            #modificació del terme independent
            b[i] = b[i] - alpha*b[k]
    x=substitucioEnrera(A,b)
    return x
 

def substitucioEnrera(A,b): 
    n=len(A)
    x = np.zeros(n)
    x[n-1] = b[n-1]/A[n-1,n-1]
    for i in range(n-2,-1,-1):
        #x[i] = (b[i] - np.dot(A[i,i+1:],x[i+1:]))/A[i,i]
         s=0
         for j in range(i+1,n):
             s=s+A[i,j]*x[j]
         x[i]=(b[i]-s)/A[i,i]    
    return x


def eliminacioGaussiana_banda(A, b,s):
    '''Eliminació Gaussiana. Es modifiquen A i b'''
    n=len(A)
    if b.size != n:
        raise ValueError("Error: les dimensions de A i b no coincideixen", b.size, n)
    for k in range(n-1):
        for i in range(k+1, min(n,k+s+1)):
            alpha = A[i,k]/A[k,k]
            #Falta posar zeros a la resta de la columna, si es vol
            #A[i,k] = alpha si es vol guardar L per LU
            A[i,k]=0
            for j in range(k+1, min(n,k+s+1)):
                A[i,j] = A[i,j] - alpha*A[k,j]
            #modificació del terme independent
            b[i] = b[i] - alpha*b[k]
    x=substitucioEnrera_banda(A,b, s)
    return x

def substitucioEnrera_banda(A,b,s):
    n=len(A)
    x=np.zeros(n)
    x[n-1]=b[n-1]/A[n-1, n-1]
    for i in range(n-2, -1, -1):
        r=0
        for j in range(i+1, min(n, i+s+1)):
            r=r+A[i,j]*x[j]
        x[i]=(b[i]-r)/A[i,i]
    return x
        
     