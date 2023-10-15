import numpy as np



def creaSistema(nRefinament,X):
    nx = nRefinament*5
    ny = nRefinament*3
    dm = (nx+1)*(ny+1)
    
    d0 = 4*np.ones(dm)
    d1 = -np.ones(dm-1)
    d2 = -np.ones(dm - (nx+1))
    A = np.diag(d0,0) + np.diag(d1,1) + np.diag(d1,-1) + np.diag(d2, -(nx+1)) + np.diag(d2,nx+1)
    for j in range(ny):
        indF = (nx+1)*j+nx 
        A[indF,indF+1] = 0
        A[indF+1,indF] = 0  

    f = np.zeros(dm)
    
    return A,f




def redueixSistema(A,f,nRefinament):
    nx = nRefinament*5
    ny = nRefinament*3
    
    nOfNodes = (nx+1)*(ny+1)
    nodesContorn = [np.arange(0, nx+1),  \
        np.arange((nx+1)*(ny),nOfNodes), \
        np.arange(nx+1,nOfNodes-nx,nx+1),\
        np.arange(2*nx+1,nOfNodes-1,nx+1)] 
    nodesContorn = np.concatenate(nodesContorn)

    nodesRadiador1 = np.arange((nx+1)*nRefinament,nOfNodes-(nx+1)*nRefinament,nx+1)
    nodesRadiador2 = nodesRadiador1 + nx
    nodesRadiador3 = np.arange( nOfNodes-nRefinament-1,nOfNodes-3*nRefinament-2,-1 )
    nodesRadiadors = np.concatenate([nodesRadiador1,nodesRadiador2,nodesRadiador3])

    nodesInterior = np.setdiff1d(np.arange(nOfNodes),nodesContorn)  
    
    u = np.zeros(nOfNodes)
    u[nodesContorn] = 10
    u[nodesRadiadors] = 30

    fred = f - A[:,nodesContorn]@u[nodesContorn]
    fred = fred[nodesInterior]
    Ared = A[nodesInterior,:]
    Ared = Ared[:,nodesInterior]
    
    return Ared,fred, nodesInterior, nodesContorn, nodesRadiadors