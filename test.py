import numpy as np

y_train = np.load("./datasets/y_train.npy")
y_test = np.load("./datasets/y_test.npy")
x_train_pca = np.load("./datasets/x_train_pca.npy")
x_test_pca = np.load("./datasets/x_test_pca.npy")

from utils.NeuralNetPy import models

network = models.Network()

from utils.NeuralNetPy import ACTIVATION, WEIGHT_INIT, layers

features_size = len(x_train_pca[0])
network.addLayer(layers.Dense(features_size))
network.addLayer(layers.Dense(318, ACTIVATION.SIGMOID, WEIGHT_INIT.GLOROT))
network.addLayer(layers.Dense(128, ACTIVATION.RELU, WEIGHT_INIT.HE))
network.addLayer(layers.Dense(2, ACTIVATION.SOFTMAX, WEIGHT_INIT.GLOROT))

from utils.NeuralNetPy import optimizers, LOSS

# Setting up the model for training
network.setup(optimizer=optimizers.Adam(0.01), loss=LOSS.BCE)

from utils.NeuralNetPy import TrainingData2dI

# Since already normalized just pass the inputs to batch with TrainData2dI
train_data = TrainingData2dI(x_train_pca, y_train)
train_data.batch(128)

from utils.NeuralNetPy import callbacks

callbacks = [callbacks.ModelCheckpoint("checkpoints", saveBestOnly=True, verbose=False)]
train_score = network.train(train_data, 50, callbacks=callbacks, progBar=True)

predictions = network.predict(x_test_pca)

predictions = np.argmax(predictions, axis=1)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

roc_auc = roc_auc_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy)
print("F1-score:", f1)
print("roc auc::", roc_auc)
