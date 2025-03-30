import torch
from collections import defaultdict

class KLDA:
    def __init__(self,num_classes,d,D,sigma,seed,device):
        self.num_classes = num_classes
        self.d = d 
        self.D = D 
        self.sigma = sigma
        self.seed = seed
        self.device = device

        torch.manual_seed(self.seed)
        self.omega = torch.normal(
            0,
            torch.sqrt(torch.tensor(2 * self.sigma,dtype=torch.float32)),
            (self.d,self.D)
        ).to(self.device)

        self.b = (torch.rand(self.D) * 2 * torch.pi).to(self.device)
        self.class_means = defaultdict(lambda: torch.zeros(self.D,device=self.device))
        self.class_counts = defaultdict(int)
        self.sigma = torch.zeros((self.D,self.D),device=self.device)
        self.sigma_inv = None
        self.class_mean_matrix = None

    def _compute_rff(self, X):
        scaling_factor = torch.sqrt(torch.tensor(2.0 / self.D, dtype=torch.float32, device=self.device))
        return scaling_factor * torch.cos(X @ self.omega + self.b)

    def batch_update(self, X, y):
        X = X.to(self.device)
        n = X.size(0)
        phi_X = self._compute_rff(X)  
        phi_X_mean = torch.mean(phi_X, dim=0)
        previous_count = self.class_counts[y]
        self.class_counts[y] += n
        self.class_means[y] = (self.class_means[y] * previous_count + phi_X_mean * n) / self.class_counts[y]
        centered_phi_X = phi_X - self.class_means[y]
        self.sigma += centered_phi_X.t() @ centered_phi_X

    def fit(self):
        self.sigma_inv = torch.pinverse(self.sigma)
        self.class_mean_matrix = torch.stack([self.class_means[i] for i in range(self.num_classes)]).to(self.device)

    def get_logits(self, x):
        x = x.to(self.device)
        phi_x = self._compute_rff(x.unsqueeze(0))  
        diff = self.class_mean_matrix - phi_x      
        logits = -torch.sum((diff @ self.sigma_inv) * diff, dim=1)  
        return logits


    def predict(self, x):
        logits = self.get_logits(x)
        return torch.argmax(logits).item()