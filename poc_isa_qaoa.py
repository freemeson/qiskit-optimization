from docplex.mp.model import Model
#from qiskit_optimization.problems.quadratic_program import QuadraticProgram

from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_optimization.translators import from_docplex_mp

mdl = Model("docplex model")
x = mdl.binary_var("x")
y = mdl.binary_var("y")
mdl.minimize(x - 2 * y)
op = from_docplex_mp(mdl)
opt_sol = -2

for sampler in [Sampler(), StatevectorSampler()]:
    qaoa = QAOA(sampler, COBYLA())
    meo = MinimumEigenOptimizer(qaoa)
    results = meo.solve(op)
    print(sampler)
    print(results)



