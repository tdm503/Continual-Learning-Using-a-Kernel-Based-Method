from klda_e import *
import torch
X_class_0 = [[1.0, 2.0],[1.5, 1.8]]
X_class_1 = [[3.0, 3.5],[3.2, 3.8]]

klda_e = KLDA_E(num_classes=2, d=2, D=3, sigma=1.0, num_ensembles=3, seed=42)

X_class_0 = torch.tensor([[1.0, 2.0], [1.5, 1.8]])
X_class_1 = torch.tensor([[3.0, 3.5], [3.2, 3.8]])

klda_e.batch_update(X_class_0, y=0)
klda_e.batch_update(X_class_1, y=1)
klda_e.fit()

x_new = torch.tensor([2.0, 2.5])
predicted_class = klda_e.predict(x_new)
print("Predicted class:", predicted_class)
