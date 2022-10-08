from ast import Num
from os import stat
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import spatial
from scipy.linalg import expm
from joblib import Parallel, delayed
import multiprocessing

#Conversion Factors and Constants
amuToKg=  1.66054e-27 
e = 1.602176634e-19
a0 = 5.29177210903e-11
Debye = 0.39*e*a0
epsilon0 = 8.8541878128e-12
hbar = 1.05457182e-34
kB = 1.38064852e-23


#Atomic/Trap Properties
Rb = {'mass': 85.467*amuToKg, 'w': 2*np.pi* np.array([84, 183, 183 ]), 'Dipole': 8000*a0*e}
KRb = {'mass': (39.0983 + 85.467)*amuToKg, 'w': 2*np.pi* np.array([110,240,240 ]), 'Dipole': 0.56*Debye }

#Function where everything happens
def System():
    NumOfKRb = 8000 #num of KRb 
    NumOfRb = 8000 #num of KRb 
    
    T = 0.5E-6 #Temperature (0.5 uK)
    fraction2Ryd = 10/8000 # ~10 rubidium rydberg atoms out of the 8000 total rubidium molecule
    t = 30 #Simulation Time (us)
    stateInit = np.array([0, 1]) #Starts at spin down
    
    decay = 0.01 #Not in use yet
    iters  = 100 #Takes an average over 10 experiments/shots
    
    SpinProgression = []
    tarray= np.linspace(0, t, 1000) #Creates time array

    SpinProgression = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(multiProcessIt)(NumOfRb, NumOfKRb, fraction2Ryd, stateInit, tarray, T) for i in range(iters))
    '''
    for i in range(iters):
        RbAtomCoord = generateAtoms(0, NumOfRb, Rb['mass']*Rb['w']**2, kB*T) #Rb coord
        KRbAtomCoord = generateAtoms(0, NumOfKRb, KRb['mass']*KRb['w']**2, kB*T) #Krb Coord

        print("Exciting to Ryd")
        RydCoord, NonRyd = exciteRydberg(RbAtomCoord, fraction2Ryd) #Returns the coordinates of the Rydberg atoms
        print("Finding Closest Neighbors")
        costNeigh = findNearestNeighbor(RydCoord, KRbAtomCoord) #Finds closest KRb neighbor for each rydber atom
        print(f"Num Of Ryd: {np.shape(RydCoord)[1]}, Distance = {costNeigh[:]} ")

        print("Calculating Hamiltonian")
        HamiltonianArray = getInteractionStrength(costNeigh) #For each distance, calculates the rydberg interaction Hamiltonian 
        print('Evolving under the hamiltonian')
        stateFinalArray, OverlapSpinUp = TimeEvolve(stateInit, HamiltonianArray, tarray) #Evolves state under this hamiltonian
        print('Data analysis')
        SpinProgression.append(OverlapSpinUp)  
        #NumOfKRb = int(NumOfKRb*np.exp(-decay))
    '''
    SpinProgAvg = np.average(SpinProgression, axis=0)
    plt.title('Rydberg Dynamics')
    plt.plot(tarray, SpinProgAvg)
    plt.xlabel('Time (us)')
    plt.ylabel('Fraction of |1, 0> of Interacting KRb Atoms')
    plt.show()

    plt.title('Rydberg Dynamics')
    plt.plot(tarray, SpinProgAvg*(fraction2Ryd*NumOfRb))
    plt.xlabel('Time (us)')
    plt.ylabel('Atom Count of |1, 0> Overall States')
    plt.show()

    plt.title('Rydberg Dynamics')
    plt.plot(tarray, SpinProgAvg*(fraction2Ryd*NumOfRb/(NumOfKRb)))
    plt.xlabel('Time (us)')
    plt.ylabel('Fraciton of |1, 0> Overall States')
    plt.show()
def multiProcessIt(NumOfRb, NumOfKRb, fraction2Ryd, stateInit, tarray, T):
    RbAtomCoord = generateAtoms(0, NumOfRb, Rb['mass']*Rb['w']**2, kB*T) #Rb coord
    KRbAtomCoord = generateAtoms(0, NumOfKRb, KRb['mass']*KRb['w']**2, kB*T) #Krb Coord

    print("Exciting to Ryd")
    RydCoord, NonRyd = exciteRydberg(RbAtomCoord, fraction2Ryd) #Returns the coordinates of the Rydberg atoms
    print("Finding Closest Neighbors")
    costNeigh = findNearestNeighbor(RydCoord, KRbAtomCoord) #Finds closest KRb neighbor for each rydber atom
    print(f"Num Of Ryd: {np.shape(RydCoord)[1]}, Distance = {costNeigh[:]} ")

    print("Calculating Hamiltonian")
    HamiltonianArray = getInteractionStrength(costNeigh) #For each distance, calculates the rydberg interaction Hamiltonian 
    print('Evolving under the hamiltonian')
    stateFinalArray, OverlapSpinUp = TimeEvolve(stateInit, HamiltonianArray, tarray) #Evolves state under this hamiltonian
    print('Data analysis')
    return OverlapSpinUp

#Generates the coordinates of a given atom number basesd on the spacial density
def generateAtoms(mu, AtomNum, TrapFreq, KbT):
    coordList = []
    #Pulls the coordinate for x, y and z
    for TrapFreqI in TrapFreq:
        sigma = np.sqrt(KbT/(TrapFreqI))
        coord = np.random.normal(mu, sigma, AtomNum)
        coordList.append(coord)
    return np.array(coordList)


def TimeEvolve(stateInit,HamiltonianArray, trange):
    stateFinalArray = []
    OverlapSpinUp = []
    tstart = np.random.rand(len(HamiltonianArray))*2
    for ind, t in enumerate(trange): #calculates the hamiltonian/Frac at Spin Up for each time step
        print(f'{round(ind/len(trange)*100, 2)}', end='\r')
        FinalState = np.array(stateInit, dtype = complex)
        Overlap = 0
        for i, H in enumerate(HamiltonianArray):
            if tstart[i] < t: 
                TimeMat = expm(-1j*H*t)
                FinalState = np.matmul(TimeMat, stateInit) 
                FinalState = FinalState/np.linalg.norm(FinalState)
                Overlap += np.abs(FinalState[0])**2

        stateFinalArray.append(FinalState)
        OverlapSpinUp.append(Overlap/len(HamiltonianArray)) #OverlapSpinUp for each t
    return  np.array(stateFinalArray), np.array(OverlapSpinUp)   
def gen3D():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    return ax
    
def exciteRydberg(AtomCoord, fraction2Ryd):
    AtomInd = np.arange(0, np.shape(AtomCoord)[1])
    population = int(fraction2Ryd*len(AtomInd)) 
    
    indofRyd = random.sample(list(AtomInd), population)
    RydCoord = np.array([AtomCoord[i][indofRyd] for i in range(np.shape(AtomCoord)[0])]) #Randomly picks Rudibium atom coord to form ryd coord
    NonRyd  = np.array([np.delete(AtomCoord[i][:], indofRyd) for i in range(np.shape(AtomCoord)[0])])
    #NewAtomCoord = AtomCoord
    return RydCoord, NonRyd
def getInteractionStrength(NeighList): #Gets interaction strength
    HamiltonianArray = [] 
    for atom in NeighList: #For each distance gets the hamiltonian
    
        distance = atom[0]
        #a3 = 5.4130e+18
        #print(a3/(1/distance**3))
        rabi = Rb['Dipole']*KRb['Dipole']/(4*np.pi*epsilon0*distance**3)/(2*np.pi*hbar)

        #rabi = Rb['Dipole']*KRb['Dipole']/(4*np.pi*epsilon0)/(2*np.pi*hbar)*a3

        Hamiltonian = rabi/2*np.array([[0, 1], [1, 0]])*1E-6
        HamiltonianArray.append(Hamiltonian)
    return HamiltonianArray


def findNearestNeighbor(RydCoord, KRbCoord):
    KRbCoord = np.transpose(KRbCoord)
    RydCoord = np.transpose(RydCoord)
    tree = spatial.KDTree(KRbCoord)
    closestNeighborList = []
    rcut = 1E-6
    for atom in RydCoord: 
        dist2Krb, indOfKRb = tree.query(atom)
        while dist2Krb < rcut:

            KRbCoord = np.delete(KRbCoord, indOfKRb, 0)
            tree = spatial.KDTree(KRbCoord)
            dist2Krb, indOfKRb = tree.query(atom)


        closestNeighborList.append([dist2Krb, indOfKRb])
    return closestNeighborList
    
#Plots Distributinon

def distribution3D(ax, AtomCoord):

    ax.scatter(AtomCoord[0], AtomCoord[1], AtomCoord[2])
    

System()
