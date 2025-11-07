from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler

feature_map = ZZFeatureMap(4)
sampler = Sampler()
kernel = QuantumKernel(feature_map=feature_map, sampler=sampler)
print("QuantumKernel initialized successfully")