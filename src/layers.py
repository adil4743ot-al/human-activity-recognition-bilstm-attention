"""Task 2: recurrent building blocks written from the equations (no nn.RNN / nn.LSTM / nn.GRU)."""
import torch
import torch.nn as nn

def uniform_param(*shape, bound):                 # trainable tensor, entries uniform in [-bound, bound]
    return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))

class RNNCell(nn.Module):
    """h_t = tanh(W x_t + U h_(t-1) + b). Only used for the discussion in Task 6.6."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        k = hidden_size ** -0.5
        self.W, self.U, self.b = (uniform_param(hidden_size, input_size, bound=k),
                                  uniform_param(hidden_size, hidden_size, bound=k), uniform_param(hidden_size, bound=k))
    def forward(self, x_t, h_prev):
        return torch.tanh(x_t @ self.W.T + h_prev @ self.U.T + self.b)

class LSTMCell(nn.Module):
    """One LSTM step. W, U, b hold the gates stacked as: input, forget, candidate, output."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        k = hidden_size ** -0.5
        self.W, self.U, self.b = (uniform_param(4 * hidden_size, input_size, bound=k),
                                  uniform_param(4 * hidden_size, hidden_size, bound=k), uniform_param(4 * hidden_size, bound=k))
    def forward(self, x_t, state):
        h_prev, c_prev = state
        i, f, g, o = (x_t @ self.W.T + h_prev @ self.U.T + self.b).chunk(4, dim=1)
        c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

class LSTM(nn.Module):
    """Run the cell over all time steps (h_0 = c_0 = 0). (B, T, D) -> (B, T, H)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
    def forward(self, x):
        h = x.new_zeros(x.shape[0], self.hidden_size)
        state, hidden_states = (h, h), []
        for x_t in x.unbind(dim=1):
            state = self.cell(x_t, state)
            hidden_states.append(state[0])
        return torch.stack(hidden_states, dim=1)

class BiLSTM(nn.Module):
    """Two independent LSTMs, one per reading direction. (B, T, D) -> (B, T, 2H)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_lstm, self.backward_lstm = LSTM(input_size, hidden_size), LSTM(input_size, hidden_size)
    def forward(self, x):
        backward = self.backward_lstm(x.flip(1)).flip(1)       # flipped back so index t matches x_t
        return torch.cat((self.forward_lstm(x), backward), dim=-1)

class Attention(nn.Module):
    """Additive attention over time: e_t = v.tanh(W_a h_t + b_a), alpha = softmax(e), c = sum alpha_t h_t."""
    def __init__(self, in_features=128, attn_size=64):
        super().__init__()
        k = attn_size ** -0.5                               # assumption: same bound as the LSTMs
        self.W_a, self.b_a, self.v = (uniform_param(attn_size, in_features, bound=k),
                                      uniform_param(attn_size, bound=k), uniform_param(attn_size, bound=k))
    def forward(self, h):
        scores = torch.tanh(h @ self.W_a.T + self.b_a) @ self.v      # (B, T)
        alpha = torch.softmax(scores, dim=1)
        return (alpha.unsqueeze(-1) * h).sum(dim=1), alpha
