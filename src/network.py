from utils.NeuralNetPy import models, layers, ACTIVATION, WEIGHT_INIT, optimizers, LOSS
import os
from utils.helpers import train_evaluate_model, filename_without_ext

network = models.Network()

network.addLayer(layers.Dense(31))
network.addLayer(layers.Dense(128, ACTIVATION.RELU, WEIGHT_INIT.GLOROT))
network.addLayer(layers.Dense(2, ACTIVATION.SOFTMAX, WEIGHT_INIT.GLOROT))

network.setup(optimizer=optimizers.Adam(0.001), loss=LOSS.BCE)

train_evaluate_model(
    network, model_name=network.getSlug(), stratified=True, shuffle=True
)
