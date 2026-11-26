"""Task 3: train HARNet with exactly the settings of the assignment."""
import os, time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from prepare import prepare, save_table1
from model import HARNet, trainable_parameters

EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_GRAD_NORM = 30, 64, 1e-3, 1.0

@torch.no_grad()
def score(model, X, y):
    model.eval()
    logits = torch.cat([model(chunk)[0] for chunk in X.split(256)])
    return F.cross_entropy(logits, y).item(), (logits.argmax(dim=1) == y).float().mean().item()

def run_epoch(model, optimizer, X, y, generator):
    model.train()
    total = 0.0
    for idx in torch.randperm(len(X), generator=generator).split(BATCH_SIZE):
        loss = F.cross_entropy(model(X[idx])[0], y[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)   # between backward and step
        optimizer.step()
        total += loss.item() * len(idx)
    return total / len(X)

os.makedirs("results", exist_ok=True)
data = prepare()
save_table1(data)
(X_tr, y_tr), (X_va, y_va) = data.train, data.val
torch.manual_seed(0)
model = HARNet()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
start_losses = score(model, X_tr, y_tr)[0], score(model, X_va, y_va)[0]
print(f"parameters {trainable_parameters(model)} | initial loss train {start_losses[0]:.4f} val {start_losses[1]:.4f} (log 6 = 1.7918)")

shuffler = torch.Generator().manual_seed(0)
history, best_loss, best_epoch, t0 = [], float("inf"), 0, time.time()
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(model, optimizer, X_tr, y_tr, shuffler)
    val_loss, val_acc = score(model, X_va, y_va)
    history.append((train_loss, val_loss, val_acc))
    if val_loss < best_loss:                               # checkpoint = lowest validation loss
        best_loss, best_epoch = val_loss, epoch
        torch.save(model.state_dict(), "results/best_model.pt")
    print(f"epoch {epoch:2d} train {train_loss:.4f} val {val_loss:.4f} val acc {val_acc:.4f}", flush=True)
minutes = (time.time() - t0) / 60

train_curve, val_curve, acc_curve = zip(*history)           # Task 3.2
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
left.plot(range(1, EPOCHS + 1), train_curve, label="training loss")
left.plot(range(1, EPOCHS + 1), val_curve, label="validation loss")
right.plot(range(1, EPOCHS + 1), acc_curve, color="tab:green")
for ax, ylabel in ((left, "cross-entropy"), (right, "validation accuracy")):
    ax.axvline(best_epoch, color="gray", ls="--")
    ax.set(xlabel="epoch", ylabel=ylabel)
left.legend()
fig.savefig("results/training_curves.png", dpi=130, bbox_inches="tight")

b = best_epoch - 1                                          # Table 2
rows = [("Trainable parameters", f"{trainable_parameters(model):,}"),
        ("Initial loss (before training), train / validation", f"{start_losses[0]:.4f} / {start_losses[1]:.4f}"),
        ("Best epoch (lowest validation loss)", best_epoch),
        ("Training / validation loss at best epoch", f"{train_curve[b]:.4f} / {val_curve[b]:.4f}"),
        ("Validation accuracy at best epoch", f"{acc_curve[b]:.4f}"),
        ("Training time and hardware", f"{minutes:.1f} min, {os.cpu_count()} CPU core(s), no GPU, torch {torch.__version__}")]
open("results/table2_training.md", "w").write("| Item | Value |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in rows))
