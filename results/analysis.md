# Task 6: Analysis

Model numbers come from the files in `results/` (`test_log.txt`, `table3_test.md`, `train_log.txt`). Numbers about the raw signals (window overlap, per-activity means, errors per volunteer) come from short one-off checks on the data and the saved predictions. Those scripts are not part of the repository.

## 6.1 Why validate by person and not by random windows?

Neighbouring windows overlap by 50%. In the training file, 6,752 of the 7,351 consecutive window pairs share exactly 64 samples. If you pick validation windows at random, almost every validation window has a neighbour in the training set that contains half of its samples. The network can then score well on validation by recognising samples it has already seen, and the validation loss says little about new data.

A second reason is that windows from one person look alike (gait, how the phone sits on the belt). The test set contains 9 volunteers the network has never seen. A validation set of 4 held-out volunteers (3, 15, 25, 30) imitates that situation, so the epoch we choose on validation is chosen on a fair signal. The cost is a noisy curve: four people give only 1,461 windows, and the validation loss moves between 0.11 and 0.31 from epoch to epoch.

## 6.2 Which two activities are confused most often?

SITTING and STANDING. On the test set 69 sitting windows were predicted as standing and 78 standing windows as sitting. That is 147 of the 236 errors (62%). The next largest group is confusion among the three walking activities (62 errors).

Task 1.3 shows why. In both still activities the body acceleration stays near zero (mean within-window standard deviation about 0.006 to 0.014 g, compared with 0.14 to 0.37 g when walking) and the gyroscope is also small. A phone on the waist stays upright in both postures and the trunk does not move. The only difference is a small tilt of the pelvis, which shows up in the gravity direction: mean `total_acc` is (0.95, 0.14, 0.15) g when sitting and (1.00, -0.16, -0.03) g when standing. The per-window ranges overlap (sitting y from -0.30 to 0.62 g, standing y from -0.36 to 0.11 g), so the tilt depends on the person and the belt position. The errors are concentrated: volunteers 2, 4, 9 and 10 account for 147 of the 148 sitting/standing errors, and volunteers 12, 13, 18 and 24 have none.

## 6.3 Which signal makes LAYING easy?

`total_acc` (the acceleration including gravity). Gravity is a constant 1 g that points down. When you stand or sit, it points along the phone's x axis, so `total_acc_x` is about 1 g (1.00 standing, 0.95 sitting). When you lie down the phone turns about 90 degrees and gravity moves to the y and z axes: `total_acc_x` is 0.07 g and the mean of y and z is 0.65 and 0.56 g. This is a static difference of about 1 g on one channel, much larger than any difference between the still postures. `body_acc` cannot show it because gravity has been removed from it, and the body is not moving when lying.

This follows from the signal statistics. I did not test it by removing channels from the trained network, so it explains why the task is easy and does not prove what the network uses.

Our LAYING accuracy is 95.0%, not the "close to 100%" the PDF mentions. All 27 errors come from one volunteer (10), whose misclassified windows have a mean `total_acc_y` of -0.96 g. The training windows of LAYING never have a per-window mean of y below -0.49 g, so this lying orientation is outside what the network saw. The other eight test volunteers have no LAYING errors.

## 6.4 Is the attention more peaked for moving or still activities?

More peaked for moving activities. The mean of max_t alpha_t is 0.225 (walking), 0.259 (upstairs) and 0.224 (downstairs), against 0.020 (sitting), 0.030 (standing) and 0.017 (laying). Flat attention would give 0.0078. For the moving activities the largest single weight is on average about 30 times the flat value, and the example plots show one to three narrow spikes of 0.1 to 0.5 weight, in the examples shown, near large peaks of the body acceleration. For the still activities the weights stay close to flat across the middle of the window.

What this suggests: when you walk, the information sits in short events (a step impact) that repeat, so the network pools a few time steps. When you sit, stand or lie, the information (the gravity direction and the lack of movement) is the same at every time step, so no moment matters more than another and the network averages. Attention weights show where the network pooled its input. They do not prove that those time steps are causally the most important ones.

One more observation from the average curves (Task 5.2): for all activities the first and last few tenths of a second get less weight (mean alpha 0.0049 over the first 10 steps and 0.0041 over the last 10, against 0.0084 over the middle). I did not test why. A possible reason is that the hidden states at the edges have context from only one side, but that is a guess. The bump of the WALKING_UPSTAIRS average near 1.2 s is also unexplained.

## 6.5 Why is a bidirectional LSTM allowed here?

The label belongs to the whole window, and the whole 2.56 s window is recorded before the network is asked for a label. Reading it backwards uses samples that exist at prediction time, so nothing from the future is leaked. The cost is a delay: you must wait 2.56 s before you can classify.

It would not be allowed when the task is to predict something that has not happened yet, or to answer before the rest of the input has arrived. Examples: forecasting tomorrow's temperature from the past week (a backward pass would read tomorrow's values), next-word prediction in a language model, or a live speech system that must output text with low delay. In these cases the backward half would see the answer or data that is not available at prediction time.

## 6.6 Why does an LSTM keep information over 128 steps better than a plain RNN, and what does clipping guard against?

In the plain RNN, h_t = tanh(W x_t + U h_{t-1} + b), so the derivative of h_t with respect to h_{t-1} is diag(1 - h_t^2) U. To send a gradient back 127 steps you multiply 127 such matrices. If they shrink the vector (tanh derivative below 1, small U), the gradient goes to zero (vanishing gradient) and the early samples have almost no influence on the loss. If U is large the product grows (exploding gradient).

In the LSTM, c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t. The direct derivative of c_t with respect to c_{t-1} is just f_t, an element-wise number between 0 and 1 that the network can learn to keep near 1. The path through the cell state uses addition and an element-wise product, with no matrix multiplication and no squashing by tanh. Over k steps the direct gradient is the product of k forget gates. If they stay close to 1, the gradient and the stored information survive. (There are also gradient paths through h and the gates, so this is the main path, not the only one.)

Gradient clipping guards against exploding gradients: when the total gradient norm exceeds 1.0, all gradients are scaled down so the norm equals 1.0, which limits how far one bad batch can move the weights. It does nothing against vanishing gradients. I did not log how often clipping was active during training, so I cannot say whether it mattered in this run.

## Notes on the results

* Test accuracy is 0.9199, inside the 90 to 93% range the PDF gives. The best validation loss (0.1100) was at epoch 22. The PDF's reference run had it around epoch 15. With four validation volunteers the curve is noisy (validation loss before epoch 22 was 0.1548 at epoch 3 and 0.1522 at epoch 16), so the exact epoch is not meaningful.
* The test windows were used once, after the epoch was chosen.
