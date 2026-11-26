| | Walk | Up | Down | Sit | Stand | Lay |
|---|---|---|---|---|---|---|
| Per-class accuracy | 0.857 | 0.979 | 0.962 | 0.839 | 0.810 | 0.968 |
| Mean of max_t alpha_t | 0.2323 | 0.2337 | 0.2685 | 0.0237 | 0.0301 | 0.0194 |

| Test accuracy |
|---|
| 0.9002 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---|---|---|---|---|---|
| WALKING | 425 | 41 | 30 | 0 | 0 | 0 |
| WALKING_UPSTAIRS | 7 | 461 | 2 | 1 | 0 | 0 |
| WALKING_DOWNSTAIRS | 5 | 7 | 404 | 4 | 0 | 0 |
| SITTING | 0 | 2 | 0 | 412 | 77 | 0 |
| STANDING | 0 | 0 | 0 | 101 | 431 | 0 |
| LAYING | 0 | 0 | 0 | 0 | 17 | 520 |
