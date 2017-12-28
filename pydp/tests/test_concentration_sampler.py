from math import log

from pydp.samplers.concentration import GammaPriorConcentrationSampler

sampler = GammaPriorConcentrationSampler(0.01, 0.01)

x = 0

for i in range(1000000):
    x = sampler.sample(x, 1, 100)

    x /= 2

    print(x)

    log(x)
