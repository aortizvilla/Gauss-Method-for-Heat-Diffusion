import numpy as np
import scipy.sparse as sps
from sistema import * 
from eliminacioGaussiana import * 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import time

# Discretització: nx divisions en la direcció horitzontal i ny en la vertical
nRefinament = 3
nx = nRefinament*5
ny = nRefinament*3

# Generem la graella de punts (matrius x,y amb les coordenades horitzontals i verticals)
# I reordenem els punts formant una matriu amb dues columnes
x,y = np.meshgrid(np.linspace(0, 5, nx+1), np.linspace(0, 3, ny+1))
nOfNodes = (nx+1)*(ny+1)
nOfInteriorNodes = (nx-1)*(ny-1)
X = np.zeros((nOfNodes,2))
X[:,0] = np.reshape(x, nOfNodes)
X[:,1] = np.reshape(y, nOfNodes)


# Construim un sistema amb tots els punts de la graella
A,f = creaSistema(nRefinament,X) 
# I el reduim per tenir en compte només els nodes interiors, on la solució no és coneguda 
Ared,fred, nodesInterior, nodesContorn, nodesRadiadors = redueixSistema(A,f,nRefinament) 

# 2) Resolem el sistema donat mitjançant el mètode de Gauss i Gauss amb Banda.
# amb un ample de banda de nx+1
Ared0=Ared.copy()
fred0=fred.copy()
Ared1=Ared.copy()
fred1=fred.copy()
ured1 = eliminacioGaussiana(Ared, fred)
ured2= eliminacioGaussiana_banda(Ared0, fred0, nx)

#Comprovem d'entrada que el sistema està ben resolt:
b=Ared1@(ured2)
sonIguals=max(abs(b-fred1))<1e-12
print('Comprovació resolució sistema OK =', sonIguals)

#I mirem si és el mateix que amb l'eliminació Gaussiana usual:
sonIguals1=(ured1==ured2).all()
print('Comprovació OK =',sonIguals1)


#3)  Mirem-ho per diferents nivells de refinament. I ens guardem:
dim=np.zeros(8) #Dimensió del sistema
timeGauss=np.zeros(8) #Temps de resolució mitjançant Gauss usual.
timeGaussbanda=np.zeros(8) # Temps de resolució mitjançant Gauss amb banda.
coef0=np.zeros(8) # Coeficients no nuls matriu original.
coef1=np.zeros(8) # Coeficients no nuls després de l'eliminació Gaussiana. 
for i in range(1,8): 
    print(i, end=": ")
    nRefinament = i
    nx = nRefinament*5
    ny = nRefinament*3

    x,y = np.meshgrid(np.linspace(0, 5, nx+1), np.linspace(0, 3, ny+1))
    nOfNodes = (nx+1)*(ny+1)
    nOfInteriorNodes = (nx-1)*(ny-1)
    X = np.zeros((nOfNodes,2))
    X[:,0] = np.reshape(x, nOfNodes)
    X[:,1] = np.reshape(y, nOfNodes)
    A,f = creaSistema(nRefinament,X) 
    Ared,fred, nodesInterior, nodesContorn, nodesRadiadors = redueixSistema(A,f,nRefinament) 
    print("Per a un nivell de refinament de", nRefinament, end='')
    dim[i]=len(Ared)
    print(" la dimensió del sistema és de", len(Ared), end='')
    coef0[i]=np.count_nonzero(Ared)
    print(". El nombre de coeficients no nuls de la matriu inicial és", coef0[i], end='')
    
    Ared0=Ared.copy()
    fred0=fred.copy()
    Ared1=Ared.copy()
    fred1=fred.copy()
   
    t_ini0=time.time()
    ured1 = eliminacioGaussiana(Ared0, fred0)
    tf0=time.time()-t_ini0
    timeGauss[i]=tf0
    print(". El temps de càlcul en resoldre el sistema mitjançant el mètode de Gauss estàndard és", timeGauss[i], end='')
   
    t_ini1=time.time()
    ured2= eliminacioGaussiana_banda(Ared1, fred1, nx)
    tf=time.time()-t_ini1
    timeGaussbanda[i]=tf
    
    print(" mentre que el temps de càlcul mitjançant el mètode adaptat és ", timeGaussbanda[i], end='')
    coef1[i]=np.count_nonzero(Ared1)
    print(". La matriu resultant té", coef1[i], end='')
    print(" coeficients no nuls.") 
    sonIguals1=(ured1==ured2).all()
    print('Comprovació OK =',sonIguals1)
    print("")

# Representem el logaritme del temps de càlcul en funció del logaritme de
# la dimensió del sistema.

plt.figure()
plt.plot(np.log10(dim[1:]), np.log10(timeGauss[1:]), label="Gauss estàndard", marker='o', color='orange')
plt.plot(np.log10(dim[1:]), np.log10(timeGaussbanda[1:]), label="Gauss adaptat a M. en banda", marker='o', color='blue')
plt.ylabel('Temps de càlcul (log)') 
plt.xlabel("Dimensió del sistema (log)") 
plt.legend()
plt.grid()
plt.savefig('gràfica1.pdf')
plt.show()


# Representem el logaritme del número de coeficients  no nuls 
# en funció del logaritme de la dimensió del sistema.

plt.figure()
plt.plot(np.log10(dim[1:]), np.log10(coef0[1:]), label="M. sistema original", marker='o', color='red')
plt.plot(np.log10(dim[1:]), np.log10(coef1[1:]), label="M. resultant del procés d'eliminació", marker='o', color='green')
plt.ylabel("Número de coeficients no nuls (log)") 
plt.xlabel("Dimensió del sistema (log)") 
plt.legend()
plt.grid()
plt.savefig('gràfica2.pdf')
plt.show()


    
