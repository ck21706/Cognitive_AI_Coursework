import neurogym as ngym
import numpy as np

# Environment
task = 'DualDelayMatchSample-v0'



dataset = ngym.Dataset(task, env_kwargs={'dt': 100})
env = dataset.env

