import torch


class LogisticRegression(torch.nn.Module):
  def __init__(self, n):
    super(LogisticRegression, self).__init__()
    self.linear = torch.nn.Linear(n, 1)

  def forward(self, x):
    logits = self.linear(x)
    return torch.sigmoid(logits).squeeze()
