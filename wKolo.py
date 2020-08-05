import numpy as np
import itertools
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit,ClassicalRegister, QuantumCircuit,execute, Aer
import math
import json
n=5
q=[]
c=[]
wKolo=[]
def parametri(N):   
    res =[ ele for ele in itertools.product([0,1,2,3], repeat = N)]
    return res
par=parametri(n)
def merX(kpom,i):
    kpom.h(i)
    return kpom

def merY(kpom,i):
    kpom.s(i)
    kpom.h(i)
    return kpom

for i in range(4**n):
    q.append(QuantumRegister(n))
    c.append(ClassicalRegister(n))
    wKolo.append(QuantumCircuit(q[i],c[i]))
    #ovde stavljamo gejtove
    wKolo[i].x(q[i][1])
    wKolo[i].ry(1.37,q[i][3])
    wKolo[i].cx(q[i][3],q[i][1])
    wKolo[i].ry(0.955,q[i][0])
    wKolo[i].ry(math.pi/4,q[i][4])
    wKolo[i].cx(q[i][1],q[i][0])
    wKolo[i].cx(q[i][3],q[i][4])
    wKolo[i].ry(-0.955,q[i][0])
    wKolo[i].ry(-math.pi/4,q[i][4])
    wKolo[i].cx(q[i][0],q[i][1])
    wKolo[i].cx(q[i][4],q[i][3])
    wKolo[i].ry(math.pi/4,q[i][2])
    wKolo[i].cx(q[i][1],q[i][2])
    wKolo[i].ry(-math.pi/4,q[i][2])
    wKolo[i].cx(q[i][2],q[i][1])
    
    wKolo[i].barrier()

#dodajemo merenja na svaki nacin
for i in range(4**n):
    for j in range(n):
        if par[i][j]==1:
            merX(wKolo[i],j)
            wKolo[i].measure(q[i][j],c[i][j])
        if par[i][j]==2:
            merY(wKolo[i],j)
            wKolo[i].measure(q[i][j],c[i][j])
        if par[i][j]==3:
            wKolo[i].measure(q[i][j],c[i][j])
#izvrsavamo kola
backend2 = Aer.get_backend('qasm_simulator')
job=[]
result=[]
counts=[]
for i in range (4**n):
    job.append(execute(wKolo[i], backend2, shots=1000))
    result.append(job[i].result())
    counts.append(result[i].get_counts(wKolo[i]))
json = json.dumps(counts)

