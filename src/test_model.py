"""Tasks 4 and 5: evaluate the saved model once on the test volunteers, then inspect the attention."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from prepare import prepare
from model import HARNet
from metrics import accuracy, confusion_matrix, per_class_accuracy
from load_data import CHANNELS

TIME = np.arange(128) / 50                                     # seconds

@torch.no_grad()
def predict(model, X):
    model.eval()
    logits, alphas = zip(*[model(chunk) for chunk in X.split(256)])
    return torch.cat(logits).argmax(dim=1).numpy(), torch.cat(alphas).numpy()

data = prepare()
names, raw_X, y = data.names, data.raw_test_X, data.test[1].numpy()
model = HARNet()
model.load_state_dict(torch.load("results/best_model.pt"))
pred, alpha = predict(model, data.test[0])                                   # Task 4
cm, per_class, acc = confusion_matrix(y, pred), per_class_accuracy(y, pred), accuracy(y, pred)
print(f"test accuracy {acc:.4f}\nconfusion matrix (rows true, columns predicted):\n{cm}\nper-class accuracy {per_class.round(3)}")
assert cm.sum() == 2947 and (cm.sum(axis=1) == np.bincount(y)).all() and np.allclose(alpha.sum(axis=1), 1, atol=1e-5)

fig, ax = plt.subplots(figsize=(7, 6))                                       # Figure 3: confusion heat-map
fig.colorbar(ax.imshow(cm, cmap="Blues"))
ax.set(xticks=range(6), xticklabels=names, yticks=range(6), yticklabels=names, xlabel="predicted activity", ylabel="true activity")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.savefig("results/confusion_matrix.png", dpi=130, bbox_inches="tight")

fig, axes = plt.subplots(2, 6, figsize=(20, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})   # Task 5.1
for k, name in enumerate(names):
    i = np.flatnonzero((y == k) & (pred == k))[0]                            # first correctly classified window
    axes[0, k].plot(TIME, raw_X[i, :, :3], label=CHANNELS[:3])
    axes[0, k].set_title(f"{name} (test window {i})", fontsize=9)
    axes[1, k].fill_between(TIME, alpha[i], color="gray")
    axes[1, k].set_xlabel("time (s)")
axes[0, 0].set_ylabel("body acc (g)")
axes[1, 0].set_ylabel("attention $\\alpha_t$")
axes[0, 0].legend(fontsize=7)
fig.savefig("results/attention_examples.png", dpi=130, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(9, 4))                                       # Task 5.2: mean attention per activity
ax.plot(TIME, np.stack([alpha[y == k].mean(axis=0) for k in range(6)]).T, label=names)
ax.axhline(1 / 128, color="gray", ls=":", label="flat attention (1/128)")
ax.set(xlabel="time (s)", ylabel="mean $\\alpha_t$")
ax.legend(fontsize=8)
fig.savefig("results/attention_average.png", dpi=130, bbox_inches="tight")

peak = np.array([alpha[y == k].max(axis=1).mean() for k in range(6)])       # Task 5.3
print("mean max alpha per activity", peak.round(4), "(flat attention = 0.0078)")

table = "| | Walk | Up | Down | Sit | Stand | Lay |\n|---|---|---|---|---|---|---|\n"           # Table 3
table += "| Per-class accuracy | " + " | ".join(f"{a:.3f}" for a in per_class) + " |\n"
table += "| Mean of max_t alpha_t | " + " | ".join(f"{p:.4f}" for p in peak) + " |\n"
table += f"\n| Test accuracy |\n|---|\n| {acc:.4f} |\n\nConfusion matrix (rows = true, columns = predicted):\n\n"
table += "| true \\ pred | " + " | ".join(names) + " |\n|---|" + "---|" * 6 + "\n"
table += "".join(f"| {name} | " + " | ".join(map(str, row)) + " |\n" for name, row in zip(names, cm))
open("results/table3_test.md", "w").write(table)
