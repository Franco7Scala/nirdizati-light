import torch.nn as nn

class Expert(nn.Module):

    def __init__(self, model):
        super(Expert, self).__init__()
        self.model = model
        if isinstance(model, nn.Model):
            self.is_torch_model = True

    def forward(self, x):
        if self.is_torch_model:
            return self.model(x)

        else:
            x = x.detach().cpu().numpy()
            output = self.model.predict(x)
            return torch.tensor(output).to(self.model.device)
