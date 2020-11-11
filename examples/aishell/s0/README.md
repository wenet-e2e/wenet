# Performance Record

## Standard E2E
| Date     | model                                    | train                                                                      | decode       | speed(min/per epoch) | CER   | Add by |
|----------|------------------------------------------|----------------------------------------------------------------------------|--------------|----------------------|-------|--------|
| /        | espnet Transformer                       | only attention loss, +speed perturb 0.9 1.1, 50 epochs                     | only decoder | /                    | 7.7   | DI WU  |
| /        | opentransformer Transformer              | only attention loss, no speed perturb, 80 epochs                           | only decoder | /                    | 6.7   | Binbin |
| 20201014 | wenet Transformer                        | joint CTC, no speed perturb, 20 epoch                                      | only decoder | 20, 1GPU             | 11.71 | Binbin |
| 20201015 | wenet Transformer                        | joint CTC, +speed perturb 0.9 1.1, 15 epoch                                | only decoder | 60, 1GPU             | 10.07 | Binbin |
| 20201018 | wenet Transformer                        | joint CTC, +speed perturb 0.9 1.1, 20 epoch, fix mask bug                  | only decoder | 60, 1GPU             | 8.13  | Binbin |
| 20201019 | wenet Transformer                        | joint CTC, +speed perturb 0.9 1.1, 50 epoch, fix mask bug                  | only decoder | /                    | 6.58  | DI WU  |
| 20201019 | wenet Transformer top5 average           | joint CTC, +speed perturb 0.9 1.1, 50 epoch, fix mask bug                  | only decoder | /                    | 6.32  | DI WU  |
| 20201025 | wenet Transformer top10 average          | joint CTC, +speed perturb 0.9 1.1, 80 epoch, fix mask bug                  | only decoder | 20, 4gpu             | 5.93  | DI WU  |
| 20201028 | wenet Transformer top10 average          | joint CTC, +speed perturb 0.9 1.1, 80 epoch, fix mask bug, accumulate grad | only decoder | 18, 4gpu             | 5.64  | DI WU  |
| 20201102 | wenet Conformer top10 average            | joint CTC, +speed perturb 0.9 1.1, 80 epoch, fix mask bug, accumulate grad | only decoder | 38, 4gpu             | 4.94  | DI WU  |
| 20201102 | wenet Conformer top10 average subsample8 | joint CTC, +speed perturb 0.9 1.1, 80 epoch, fix mask bug, accumulate grad | only decoder | /,  4gpu             | 5.16  | DI WU  |
| 20201102 | wenet Conformer top10 average subsample6 | joint CTC, +speed perturb 0.9 1.1, 80 epoch, fix mask bug, accumulate grad | only decoder | 19, 6gpu             | 5.06  | DI WU  |


## Unified Dynamic chunk

On conformer(causal convolution), configure: conf/train_unified_conformer.yaml

| decoding mode/chunk size | full | 16   | 8    | 4    | 1    |
|--------------------------|------|------|------|------|------|
| attention decoder        | 5.82 | 5.99 | 6.13 | 6.29 | 6.6  |
| ctc greedy search        | 5.51 | 6.23 | 6.57 | 6.92 | 7.83 |
| ctc prefix beam search   | 5.53 | 6.23 | 6.57 | 6.92 | 7.83 |
| attention rescoring      | 5.12 | 5.53 | 5.72 | 5.82 | 6.36 |

