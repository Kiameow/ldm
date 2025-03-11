import torch.nn as nn
import torch

class DummyTextEmbedder(nn.Module):
    def __init__(self, d_cond=768):
        super(DummyTextEmbedder, self).__init__()
        self.d_cond = d_cond

    def forward(self, texts):
        # Assume texts is a list (or similar) of prompts.
        batch_size = len(texts) if isinstance(texts, list) else texts.shape[0]
        # Return a dummy context tensor filled with zeros.
        return torch.zeros(batch_size, self.d_cond)
