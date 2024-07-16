import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

# Step 1: Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)
data = pd.DataFrame(X, columns=["feature1", "feature2"])
data["label"] = y


# Step 2: Create Normal Mini-Batches
def create_normal_minibatches(data, batch_size):
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    mini_batches = [
        shuffled_data[i : i + batch_size]
        for i in range(0, len(shuffled_data), batch_size)
    ]
    return mini_batches


# Step 3: Create Stratified Mini-Batches
def create_stratified_minibatches(data, batch_size):
    skf = StratifiedKFold(n_splits=int(np.ceil(len(data) / batch_size)))
    mini_batches = []
    for _, test_index in skf.split(data, data["label"]):
        mini_batches.append(data.iloc[test_index])
    return mini_batches


# Step 4: Visualize the Distribution
def plot_batch_distribution(mini_batches, title):
    batch_sizes = [
        batch["label"].value_counts(normalize=True) for batch in mini_batches
    ]
    batch_df = pd.DataFrame(batch_sizes)
    batch_df.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.xlabel("Batch Number")
    plt.ylabel("Class Proportion")
    plt.show()


# Create mini-batches
batch_size = 50
normal_batches = create_normal_minibatches(data, batch_size)
stratified_batches = create_stratified_minibatches(data, batch_size)

# Plot distributions
plot_batch_distribution(normal_batches, "Normal Mini-Batches")
plot_batch_distribution(stratified_batches, "Stratified Mini-Batches")
