"""
example 1 chapter 1 of Eliasmith's How to Build a Brain
single neuron model based on a script by s72sue available at
https://github.com/s72sue/Nengo2-Tutorials/tree/master/chapter1
"""
import numpy as np  # numerical methods
import nengo
import nengo_gui
# nengo itself
model = nengo.Network(label='Single Neuron')
with model:
    cos = nengo.Node(lambda t: np.cos(16 * t))
    neuron = nengo.Ensemble(1, dimensions=1, encoders=[[1]])
    nengo.Connection(cos, neuron)


nengo_gui.GUI(__file__).start()
