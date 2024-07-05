from qiskit_algorithms.optimizers import COBYLA

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    HGate,
    PhaseGate,
    QAOAAnsatz,
    RXGate,
    TdgGate,
    TGate,
    XGate,
)
# Pre-defined ansatz circuit, operator class and visualization tools
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_distribution

# IBM Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Sampler, Session


from qiskit.primitives import Estimator as LocalEstimator
from qiskit.primitives import Sampler as LocalSampler

# SciPy minimizer routine
from scipy.optimize import minimize




# local estimator and sampler
from qiskit.primitives import Estimator as LocalEstimator
from qiskit.primitives import Sampler as LocalSampler

# this is older qiskit
#from qiskit.providers.fake_provider import FakeBoeblingenV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout  # needed only for qiskit <=0.44
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    CXCancellation,
    InverseCancellation,
    PadDynamicalDecoupling,
    TrivialLayout,
    UnitarySynthesis,
)

# using a fake provider
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService, Sampler, Session
from scipy.optimize import minimize

def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    sense = 1
    cost = (
        #sense *
        estimator.run(ansatz, hamiltonian, parameter_values=params)
        .result()
        .values[0]
    )
    return cost

hamiltonian = SparsePauliOp.from_list([("IIIZZ", 1), ("IIZIZ", 1), ("IZIIZ", 1), ("ZIIIZ", 1)])
ansatz = QAOAAnsatz(hamiltonian, reps=2)
num_var = 5

class QUBOQAOAAnzats:
    def __init__(self):

        # if self.params.sense == OptimizationSense.MAXIMIZE.value:
        #     self.sense = -1.0
        # else:
        self.sense = 1.0

    # def _build_hamiltonian(self, backend_qubits=None):
    #     self.q = QAOACircuitBuilder(self._qubo, self.params.sense, self._qubo.n)
    #     return self.q.hermetian_hamilton_operator_sparsepauli(backend_qubits)

    def _create_local_backend_future(self):
        # This is a placeholder for qiskit 1.0, was tested on small samples
        # it could be we may need ancillary qubits, QAOAansatz may report it
        self._runtime_backend = GenericBackendV2(num_qubits=num_var)
        self._pass_manager = generate_preset_pass_manager(3, self._runtime_backend)

        dd_sequence = [XGate(), XGate()]

        scheduling_pm = PassManager(
            [
                ALAPScheduleAnalysis(target=self._runtime_backend.target),
                PadDynamicalDecoupling(
                    target=self._runtime_backend.target, dd_sequence=dd_sequence
                ),
            ]
        )
        inverse_gate_list = [
            HGate(),
            (RXGate(np.pi / 4), RXGate(-np.pi / 4)),
            (PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4)),
            (TGate(), TdgGate()),
        ]
        logical_opt = PassManager(
            [
                CXCancellation(),
                InverseCancellation(inverse_gate_list),
            ]
        )

        # Add pre-layout stage to run extra logical optimization
        self._pass_manager.pre_layout = logical_opt
        # Set scheduling stage to custom pass manager
        self._pass_manager.scheduling = scheduling_pm

    # def _set_remote_backend(self):
    #     target = self._runtime_backend.target
    #     pm = generate_preset_pass_manager(target=target, optimization_level=3)
    #     pm.scheduling = PassManager(
    #         [
    #             ApplyLayout(),
    #             ALAPScheduleAnalysis(durations=target.durations()),
    #             PadDynamicalDecoupling(
    #                 durations=target.durations(),
    #                 dd_sequence=[XGate(), XGate()],
    #                 pulse_alignment=target.pulse_alignment,
    #             ),
    #         ]
    #     )

    def _create_local_backend(self):
        self._runtime_backend = FakeBoeblingenV2()
        self._pass_manager = generate_preset_pass_manager(3, self._runtime_backend)
        # #in case of a real backend, we need a dd_sequence in paddynamicaldecoupling
        # dd_sequence = [XGate(), XGate()]
        scheduling_pm = PassManager(
            [
                TrivialLayout(coupling_map=self._runtime_backend.coupling_map),
                ApplyLayout(),
                UnitarySynthesis(synth_gates=["cx"]),
            ]
        )
        inverse_gate_list = [
            HGate(),
            (RXGate(np.pi / 4), RXGate(-np.pi / 4)),
            (PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4)),
            (TGate(), TdgGate()),
        ]
        logical_opt = PassManager(
            [
                CXCancellation(),
                InverseCancellation(inverse_gate_list),
            ]
        )

        # Add pre-layout stage to run extra logical optimization
        self._pass_manager.pre_layout = logical_opt
        # Set scheduling stage to custom pass manager
        self._pass_manager.scheduling = scheduling_pm

    def _create_qaoa_circuit_future(self, qaoa_rep=2, warmstart=None):
        # ##this is untested, but essentially we want this in qiskit 1.0
        backend_qubits = self._runtime_backend.num_qubits
        hamiltonian = self._build_hamiltonian(backend_qubits)
        self._ansatz = QAOAAnsatz(hamiltonian, reps=qaoa_rep)
        # ##a new anzatz with a different level of approximation for the exp(iH)
        self._ansatz = self._ansatz.decompose(reps=3)
        self._ansatz_pm = self._pass_manager.run(self._ansatz)
        # ##in qiskit 1.0 and qiskit-runtim >0.2 we apply the layout here and not in the pass manager
        self._hamiltonian_with_layout = hamiltonian.apply_layout(self._ansatz.layout)

    def _create_qaoa_circuit(self, qaoa_rep=2, warmstart=None):
        backend_qubits = self._runtime_backend.num_qubits
        hamiltonian = self._build_hamiltonian(backend_qubits)
        self._ansatz = QAOAAnsatz(hamiltonian, reps=qaoa_rep)
        # ##a new anzatz with a different level of approximation for the exp(iH)
        self._ansatz = self._ansatz.decompose(reps=3)
        self._ansatz_pm = self._pass_manager.run(self._ansatz)
        # ##layout is applied in the pass manager, in qiskit 1.0 this is a hamiltonian.apply_layout(self._ansatz.layout)
        self._hamiltonian_with_layout = hamiltonian
 
    def _create_qaoa_circuit_from_hamiltonian(self, qaoa_rep=2, warmstart=None):
        backend_qubits = self._runtime_backend.num_qubits
        hamiltonian = self.hamiltonian #self._build_hamiltonian(backend_qubits)
        self._ansatz = QAOAAnsatz(hamiltonian, reps=qaoa_rep)
        # ##a new anzatz with a different level of approximation for the exp(iH)
        self._ansatz = self._ansatz.decompose(reps=3)
        self._ansatz_pm = self._pass_manager.run(self._ansatz)
        # ##layout is applied in the pass manager, in qiskit 1.0 this is a hamiltonian.apply_layout(self._ansatz.layout)
        self._hamiltonian_with_layout = hamiltonian

    def solve_local_scipy(self, hamiltonian):
        self.hamiltonian = hamiltonian
#        self._qubo = qubo_data
        self._create_local_backend_future()
        self._create_qaoa_circuit_from_hamiltonian()

        def cost_func(params, ansatz, hamiltonian, estimator):
            """Return estimate of energy from estimator

            Parameters:
                params (ndarray): Array of ansatz parameters
                ansatz (QuantumCircuit): Parameterized ansatz circuit
                hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
                estimator (Estimator): Estimator primitive instance

            Returns:
                float: Energy estimate
            """
            cost = (
                self.sense
                * estimator.run(ansatz, hamiltonian, parameter_values=params)
                .result()
                .values[0]
            )
            return cost

        estimator = LocalEstimator(options={"shots": int(1e3)})
        x0 = 2 * np.pi * np.random.rand(self._ansatz_pm.num_parameters)
        res = minimize(
            cost_func,
            x0,
            args=(self._ansatz_pm, self._hamiltonian_with_layout, estimator),
            method="COBYLA",
        )
        qc = self._ansatz.assign_parameters(res.x)
        qc_meas = QuantumCircuit(
            self._runtime_backend.num_qubits, num_var
        ).compose(qc)
        qc_meas.measure(
            range(
                self._runtime_backend.num_qubits - 1,
                self._runtime_backend.num_qubits - num_var - 1,
                -1,
            ),
            range(num_var),
        )
        qc_ibm = self._pass_manager.run(qc_meas)
        sampler = LocalSampler(options={"shots": int(1e3)})
        qdist = sampler.run(qc_ibm).result().quasi_dists[0]

        return qdist


    def solve_local_qiskit_optim(self, hamiltonian):
        self.hamiltonian = hamiltonian
#        self._qubo = qubo_data
        self._create_local_backend_future()
        self._create_qaoa_circuit_from_hamiltonian()

        def cost_func(params, ansatz, hamiltonian, estimator):
            """Return estimate of energy from estimator

            Parameters:
                params (ndarray): Array of ansatz parameters
                ansatz (QuantumCircuit): Parameterized ansatz circuit
                hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
                estimator (Estimator): Estimator primitive instance

            Returns:
                float: Energy estimate
            """
            cost = (
                self.sense
                * estimator.run(ansatz, hamiltonian, parameter_values=params)
                .result()
                .values[0]
            )
            return cost

        estimator = LocalEstimator(options={"shots": int(1e3)})
        x0 = 2 * np.pi * np.random.rand(self._ansatz_pm.num_parameters)

        res = minimize(
            cost_func,
            x0,
            args=(self._ansatz_pm, self._hamiltonian_with_layout, estimator),
            method="COBYLA",
        )

        qc = self._ansatz.assign_parameters(res.x)
        qc_meas = QuantumCircuit(
            self._runtime_backend.num_qubits, num_var
        ).compose(qc)
        qc_meas.measure(
            range(
                self._runtime_backend.num_qubits - 1,
                self._runtime_backend.num_qubits - num_var - 1,
                -1,
            ),
            range(num_var),
        )
        qc_ibm = self._pass_manager.run(qc_meas)
        sampler = LocalSampler(options={"shots": int(1e3)})
        qdist = sampler.run(qc_ibm).result().quasi_dists[0]

        return qdist


    # def solve_remote(self, qubo_data: QuboData, simulator=True):
    #     self._qubo = qubo_data
    #     instance = (
    #         "ibm-q-startup/quantagonia/hybridsolver" if simulator else "ibm-q/open/main"
    #     )
    #     service = QiskitRuntimeService(
    #         channel="ibm_quantum", instance=instance, token=environ["QISKIT_IBM_TOKEN"]
    #     )
    #     if simulator:
    #         self._remote_backend = service.get_backend("ibmq_qasm_simulator")
    #         self._create_local_backend()
    #     else:
    #         self._runtime_backend = service.least_busy(
    #             operational=True, simulator=False
    #         )

    #     self._create_qaoa_circuit()

    #     def cost_func(params, ansatz, hamiltonian, estimator):
    #         """Return estimate of energy from estimator

    #         Parameters:
    #             params (ndarray): Array of ansatz parameters
    #             ansatz (QuantumCircuit): Parameterized ansatz circuit
    #             hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
    #             estimator (Estimator): Estimator primitive instance

    #         Returns:
    #             float: Energy estimate
    #         """
    #         cost = (
    #             self.sense
    #             * estimator.run(ansatz.assign_parameters(params), hamiltonian)
    #             .result()
    #             .values[0]
    #         )
    #         return cost

    #     with Session(backend=self._remote_backend) as session:
    #         estimator = Estimator(session=session, options={"shots": int(1e3)})
    #         x0 = 2 * np.pi * np.random.rand(self._ansatz_pm.num_parameters)
    #         res = minimize(
    #             cost_func,
    #             x0,
    #             args=(self._ansatz_pm, self._hamiltonian_with_layout, estimator),
    #             method="COBYLA",
    #         )
    #         sampler = Sampler(session=session, options={"shots": int(1e3)})
    #         qc = self._ansatz.assign_parameters(res.x)
    #         qc_meas = QuantumCircuit(
    #             self._runtime_backend.num_qubits, self._qubo.n
    #         ).compose(qc)
    #         qc_meas.measure(
    #             range(
    #                 self._runtime_backend.num_qubits - 1,
    #                 self._runtime_backend.num_qubits - self._qubo.n - 1,
    #                 -1,
    #             ),
    #             range(self._qubo.n),
    #         )
    #         qc_ibm = self._pass_manager.run(qc_meas)
    #         qdist = sampler.run(qc_ibm).result().quasi_dists[0]

    #     return qdist


qa = QUBOQAOAAnzats()
qa.solve_local_scipy(hamiltonian)