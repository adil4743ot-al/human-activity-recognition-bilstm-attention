"""Task 2.6: BiLSTM -> attention -> Linear(128, 6)."""
import torch
import torch.nn as nn
from layers import BiLSTM, Attention

class HARNet(nn.Module):
    def __init__(self, n_signals=9, hidden=64, n_classes=6):
        super().__init__()
        self.encoder = BiLSTM(n_signals, hidden)
        self.attend = Attention(2 * hidden, 64)
        self.classifier = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):                                   # x: (B, 128, 9)
        context, alpha = self.attend(self.encoder(x))       # (B, 128), (B, 128)
        return self.classifier(context), alpha              # logits (B, 6), weights (B, 128)

def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    torch.manual_seed(0)
    print("trainable parameters:", trainable_parameters(HARNet()), "= 2*4*64*(9+64+1) + (64*128+64+64) + (128*6+6)")
