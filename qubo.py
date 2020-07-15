from pyqubo import Spin
import neal

s1, s2, s3, s4, s5, s6 = Spin("s1"), Spin("s2"), Spin("s3"), Spin("s4"), Spin("s5"), Spin("s6")
H = -(s1 - s2 + s3*s4 - s4*s5)
model = H.compile()
qubo, offset = model.to_qubo()

sampler = neal.SimulatedAnnealingSampler()
response = sampler.sample_qubo(qubo)
print (response)