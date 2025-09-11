import torch
import torch.nn as nn
import torch.nn.functional as F


class MOEHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2,
                 num_experts: int = 4, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        gates = self.gate(x)                              
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1) 
        return (gates.unsqueeze(-1) * expert_outs).sum(1)              