import numpy as np
import itertools
import json
import matplotlib.pyplot as plt
import scipy.linalg as la


n=5
with open('countsp=0.15.txt') as json_file:
    counts = json.load(json_file)
M = np.load('matricaMp=0.15.npy')
invM = la.inv(M)

s= [ele for ele in itertools.product(['0','1'], repeat = n)]
stanjastring=[]
for i in s:
    pom=''
    pom=i[0]
    for j in range(1,len(i)):
        pom+=i[j]
    stanjastring.append(pom)

for c in counts:
    for red in stanjastring:
        if (red in c)==False: 
            c.update({red:0})

def vratiLepDit(Not_Ordered,key_order):
    ordered_dict_items = [(k, Not_Ordered[k]) for k in key_order]
    return dict((x, y) for x, y in ordered_dict_items)

matricaC=[]
noviCounts=[]

def vratiMatricu(lista):
    matrica = np.ones((len(lista),1))
    for i in range (0,len(lista)):
        matrica[i][0] = lista[i]
    return matrica
    

for c in counts:
    c1=vratiLepDit(c, stanjastring)
    vektor = list(c1.values())
    vektor = vratiMatricu(vektor)
   # matricaC.append(vektor)   
    cMit = np.dot(invM,vektor)   
    noviCounts.append(dict(zip(stanjastring, cMit)))
counts = noviCounts

def parametri(N):   
    res =[ ele for ele in itertools.product([0,1,2,3], repeat = N)]
    return res
par=parametri(n)

def plotMapu(matrica, naslov):
    fig, ax = plt.subplots(figsize=(8,8))
    cm=ax.matshow(matrica, cmap=plt.cm.Blues)
    no_labels = len(matrica) # how many labels to see on axis x
    positions = np.arange(0,no_labels) # pixel count at label position
    labels = stanjastring # labels you want to see
    plt.xticks(positions, labels,rotation=90)
    plt.yticks(positions, labels)
#    for i in range(no_labels):
#       for j in range(no_labels):
#          c = matrica[j,i]
    plt.colorbar(cm)
    plt.title(naslov,y=-0.1)
    plt.savefig(naslov +'.png')
    plt.show()

#Ova funkcija definise da li je u sumi svih verovatnoca + ili -
#prva lista ima predsstavlja indekse parametra, a drugi koja je kombinacija 0 i 1
def znak(i,s):
    rez=1
    for j in range(n):
        if s[j]=='1':
            rez*=-1
    return rez

#ovde se racunaju Stoksovi
S=np.zeros(4**n)
for p in range(4**n):
    if p==0:
        S[0]=1
    else:
        for j in range(2**n):
            S[p]+=znak(par[p],s[j])*counts[p][stanjastring[j]]/10000
            #ovde sam kao uzela da zaokruzim stoksove par, nemamm pojma koliko je to pametno tako da se radi
            #S[p]=round(S[p],2)

#ovde tenzorski mnozimo sigme, ovaj deo koda je specifican za broj qubita!!
j = complex(0,1)
matrica = [[[1,0],[0,1]],[[0,1],[1,0]],[[0,-j],[j,0]],[[1,0],[0,-1]]]
tenzorski=[]
for red in par:
    tenzorski.append(np.kron(matrica[red[0]],np.kron(matrica[red[1]],np.kron(matrica[red[2]],np.kron(matrica[red[3]],matrica[red[4]])))))

densityMatrix=np.zeros((2**n,2**n),dtype=np.complex_)
for i in range (len(S)):
    densityMatrix+=np.dot(S[i],tenzorski[i])
densityMatrix=np.dot(1/(2**n),densityMatrix)
plotMapu(densityMatrix.real,'real part - popravljen sum, p=0.15')
plotMapu(densityMatrix.imag,'imag part - popravljen sum, p=0.15')
print('done')
