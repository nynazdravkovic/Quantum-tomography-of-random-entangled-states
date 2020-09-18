# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:59:00 2020

@author: Nina
"""
from qiskit.ignis.verification import process_tomography_circuits, ProcessTomographyFitterTomographyFitter
from qiskit import(
  QuantumCircuit,
  execute,
  BasicAer,QuantumRegister,
  ClassicalRegister)
import math

q = QuantumRegister(5)
c =ClassicalRegister(5)
wKolo = QuantumCircuit(q,c)

wKolo.x(q[1])
wKolo.ry(1.37,q[3])
wKolo.cx(q[3],q[1])

wKolo.ry(0.955,q[0])
wKolo.ry(math.pi/4,q[4])
wKolo.cx(q[1],q[0])
wKolo.cx(q[3],q[4])

wKolo.ry(-0.955,q[0])
wKolo.ry(-math.pi/4,q[4])
wKolo.cx(q[0],q[1])
wKolo.cx(q[4],q[3])

wKolo.ry(math.pi/4,q[2])
wKolo.cx(q[1],q[2])
wKolo.ry(-math.pi/4,q[2])
wKolo.cx(q[2],q[1])

stanja= state_tomography_circuits(wKolo,q)
counts=[]
job = execute( stanja, BasicAer.get_backend('qasm_simulator'),shots=1000)
qpt_tomo = StateTomographyFitter(job.result(), stanja)
print(qpt_tomo.data)
