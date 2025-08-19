from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from base import BaseModel

"""class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)"""

class MaskedLinear(BaseModel):
    def __init__(self, input_size, output_size, topK, dropout_rate=0.5):
        super(MaskedLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.topK = topK
        self.dropout_rate = dropout_rate
        self.linear = nn.Linear(input_size, output_size)
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_in")
        self.dropout = nn.Dropout(self.dropout_rate)
        self.mask = nn.Parameter(torch.ones((self.input_size,), dtype=torch.uint8),
                                 requires_grad=False)

    def open_gates(self):
        # Create a boolean mask initialized with all elements set to 1 (True)
        #print("***Gates open!")
        self.mask = nn.Parameter(torch.ones((self.input_size,), dtype=torch.uint8),
                                 requires_grad=False)

    def close_gates(self):
        #print("***Gates closed!")
        weight = self.linear.weight.clone()
        weight = torch.mean(torch.abs(weight), dim=0, keepdim=True)

        # Flatten the weight tensor and calculate the k-th value
        sorted_weights, _ = torch.sort(torch.abs(weight.flatten()), descending=True)
        thr = sorted_weights[self.topK - 1]  # indexes in sorted_weights start from zero...

        # Create a mask based on the threshold
        if self.output_size > 1:
            mask_temp = (torch.abs(weight) >= thr).type(torch.uint8).squeeze()
            self.mask = nn.Parameter(mask_temp, requires_grad=False)
        else:
            self.mask = nn.Parameter((torch.abs(weight) >=
                                      thr).type(torch.uint8).squeeze(),
                                     requires_grad=False)


    def freeze(self):
        self.linear.requires_grad_(False)

    def unfreeze(self):
        self.linear.requires_grad_(True)

    def forward(self, x):
        # Apply mask
        x = self.dropout(x)
        x = x * self.mask
        return self.linear(x)

    def predict(self, x):
        x = x * self.mask
        return self.linear(x)


MoEResult = namedtuple('MoEResult', ['weighted_outputs', 'experts_outputs', 'gate_probs'])
MoEPrediction = namedtuple('MoEPrediction', ['moe_prediction', 'choosen_expert', 'experts_prediction'])


class MoE(BaseModel):
    def __init__(self, input_size, num_experts, topk, dropout_rate=0.1, temperature=0.5):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([MaskedLinear(input_size, 1, topk, dropout_rate)
                                      for _ in range(num_experts)])
        self.gate = MaskedLinear(input_size, num_experts, topk, dropout_rate)
        self.temperature = temperature

    def open_gates(self):
        for expert in range(self.num_experts):
           self.experts[expert].open_gates()
        self.gate.open_gates()

    def close_gates(self):
        for expert in range(self.num_experts):
           self.experts[expert].close_gates()
        self.gate.close_gates()

    def freeze_experts(self):
        for expert in range(self.num_experts):
            self.experts[expert].freeze()

    def unfreeze_experts(self):
        for expert in range(self.num_experts):
            self.experts[expert].unfreeze()

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=1)
        #gate_probs = F.gumbel_softmax(gate_scores, tau=self.temperature, hard=False, dim=1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weighted_outputs = torch.bmm( gate_probs.unsqueeze(1), expert_outputs ).squeeze(1)

        return MoEResult(weighted_outputs, expert_outputs, gate_probs) 

    def predict(self, x):
        gate_logits = self.gate.predict(x)
        gate_probs = F.softmax(gate_logits, dim=1)
        choosen_expert = torch.argmax(gate_probs, dim=1)
        experts_prediction = torch.stack([F.sigmoid(expert(x)) for expert in self.experts], dim=1)
        result = []
        for record_i, expert_j in enumerate(choosen_expert):
            result.append( F.sigmoid( self.experts[expert_j].predict(x[record_i].view(1,-1)).squeeze() ) )
        #return F.sigmoid(torch.stack(result, dim=0))
        return MoEPrediction( torch.stack(result, dim=0), choosen_expert, experts_prediction )
