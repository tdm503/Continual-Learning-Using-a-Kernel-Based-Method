from klda import *
import torch

num_classes = 2
d = 2  
D = 3   
sigma = 1.0

klda_model = KLDA(num_classes=num_classes, d=d, D=D, sigma=sigma,seed=1,device='cpu')

X_class_0 = torch.tensor([[1.0, 2.0], [1.5, 1.8]])
X_class_1 = torch.tensor([[3.0, 3.5], [3.2, 3.8]])

klda_model.batch_update(X_class_0, y=0)
klda_model.batch_update(X_class_1, y=1)

klda_model.fit()    

x_new = torch.tensor([2.0, 2.5])

logits = klda_model.get_logits(x_new)
predicted_class = klda_model.predict(x_new)

print("Logits:", logits)
print("Predicted class:", predicted_class)


