# Human activity recognition with a BiLSTM and attention

A recurrent network written from its equations in PyTorch. It reads one 2.56 s window of phone motion (128 time steps, 9 signals) and predicts one of six activities. The attention layer shows which moments of the window the network used.

Data: UCI "Human Activity Recognition Using Smartphones" (raw signals, not the 561 hand-made features).

**Result on the 9 unseen test volunteers: 91.99% accuracy (2,711 of 2,947 windows).** Network: 46,982 trainable parameters.

![front end](results/frontend_screenshot.png)

## Run everything

```bash
pip install -r requirements.txt

# 1. data: download from UCI, unzip into data/ (and unzip "UCI HAR Dataset.zip" inside it)
#    https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
python src/load_data.py          # prints shapes
python src/plot_windows.py       # results/example_windows.png

# 2. checks of the layers and the parameter count (with the hand calculation)
python src/model.py

# 3. train (30 epochs, about 5 minutes on one CPU core), saves results/best_model.pt
python src/train.py

# 4. test once, attention plots, tables
python src/test_model.py

# 5. front end (opens in the browser)
streamlit run app/app.py
```

Run all commands from the repository root. `python src/prepare.py` prints the Task 1 numbers (shapes, labels, split, standardisation).

## Front end

In the sidebar choose "Test window" (an activity and a window number from the test set) or "CSV upload" (128 rows and 9 columns, no header, raw units, columns in this order: body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z). The page shows the predicted activity, the signals, a bar chart with the six probabilities and the attention weights over time. All of it comes from the trained network in `results/best_model.pt`.

## Layout

| Path | Content |
|---|---|
| `data/` | unzipped dataset (not committed) |
| `src/load_data.py`, `src/plot_windows.py` | the given loader and plotting helper |
| `src/prepare.py` | Task 1: relabel, split by person, standardise, tensors |
| `src/layers.py` | `RNNCell`, `LSTMCell`, `LSTM`, `BiLSTM`, `Attention` |
| `src/model.py` | `HARNet` |
| `src/train.py` | Task 3 |
| `src/metrics.py` | `accuracy`, `confusion_matrix`, `per_class_accuracy` |
| `src/test_model.py` | Tasks 4 and 5 |
| `results/` | figures, tables, logs, `analysis.md`, `best_model.pt` |
| `app/app.py` | Streamlit front end |
| Task 7.2 and 7.3 texts |

## Setup details

* Validation volunteers: 3, 15, 25, 30. Windows: train 5,891, validation 1,461, test 2,947. Volunteers: 17 / 4 / 9.
* Standardisation uses the mean and standard deviation of the training windows only (`results/table1_data.md`).
* Training: cross-entropy, Adam with learning rate 1e-3, batch size 64, 30 epochs, shuffle every epoch, gradient clipping at total norm 1.0, `torch.manual_seed(0)`. The parameters of the epoch with the lowest validation loss (epoch 22) are used for testing.
* Assumption: the attention parameters `W_a`, `b_a`, `v` use the same uniform initialisation as the recurrent weights, with bound 1/sqrt(64). The final `nn.Linear` uses the PyTorch default.
* Reported training loss per epoch is the average of the batch losses during that epoch (weights updating as it runs). Validation loss is computed after the epoch with the model in evaluation mode.
* Training ran on one CPU core, no GPU, 5.7 minutes. Results can differ slightly on other hardware or torch versions.

## Results

See `results/table1_data.md`, `results/table2_training.md`, `results/table3_test.md` and `results/analysis.md`.

| | Walk | Up | Down | Sit | Stand | Lay |
|---|---|---|---|---|---|---|
| Per-class accuracy | 0.911 | 0.983 | 0.979 | 0.857 | 0.853 | 0.950 |
| Mean of max_t alpha_t | 0.2247 | 0.2591 | 0.2236 | 0.0202 | 0.0301 | 0.0165 |

Almost all errors are SITTING and STANDING mistaken for each other (147 of 236). All 27 LAYING errors come from one volunteer (see `results/analysis.md`, 6.3).

## Team

Group members, blog link and LinkedIn link: add here.
