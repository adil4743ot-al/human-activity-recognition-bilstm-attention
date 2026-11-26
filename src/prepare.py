"""Task 1: relabel, split by volunteer, standardise with training statistics, make tensors."""
from types import SimpleNamespace
import numpy as np
import torch
from load_data import load_raw, CHANNELS

VAL_SUBJECTS = [3, 15, 25, 30]

def prepare(data_dir="data", verbose=False):
    raw = load_raw(data_dir)
    train, test = raw["train"], raw["test"]
    labels = train["y"] - 1                                   # 1..6 -> 0..5
    in_val = np.isin(train["subject"], VAL_SUBJECTS)
    fit_X = train["X"][~in_val]
    mean, std = fit_X.mean(axis=(0, 1)), fit_X.std(axis=(0, 1))   # training windows only

    def pack(X, y):
        return torch.tensor((X - mean) / std, dtype=torch.float32), torch.tensor(y)

    data = SimpleNamespace(
        train=pack(fit_X, labels[~in_val]), val=pack(train["X"][in_val], labels[in_val]),
        test=pack(test["X"], test["y"] - 1), raw_test_X=test["X"], mean=mean, std=std,
        names=raw["activity_names"],
        volunteers={"train": np.unique(train["subject"][~in_val]).tolist(),
                    "val": np.unique(train["subject"][in_val]).tolist(),
                    "test": np.unique(test["subject"]).tolist()})
    if verbose:
        print("shapes:", {s: {k: v.shape for k, v in raw[s].items()} for s in ("train", "test")})
        print("windows per activity, train:", np.bincount(labels), "test:", np.bincount(test["y"] - 1))
        print("windows train/val/test:", *(len(getattr(data, s)[1]) for s in ("train", "val", "test")), "| volunteers:", {k: len(v) for k, v in data.volunteers.items()})
    return data

def save_table1(data, path="results/table1_data.md"):
    windows = " / ".join(str(len(getattr(data, s)[1])) for s in ("train", "val", "test"))
    people = " / ".join(str(len(data.volunteers[s])) for s in ("train", "val", "test"))
    lines = ["| Item | Value |", "|---|---|", f"| Train / validation / test windows | {windows} |",
             f"| Train / validation / test volunteers | {people} |", "",
             "Mean and standard deviation of each channel (training windows only):", "",
             "| Channel | Mean | Std |", "|---|---|---|"]
    lines += [f"| {c} | {m:.5f} | {s:.5f} |" for c, m, s in zip(CHANNELS, data.mean, data.std)]
    open(path, "w").write("\n".join(lines) + "\n")
    np.savez("results/norm_stats.npz", mean=data.mean, std=data.std)    # used by the front end

if __name__ == "__main__":
    save_table1(prepare(verbose=True))
