# Performance Record

| Date     | model                           | train                                                                      | decode       | speed(min/per epoch) | CER   | Add by |
|----------|---------------------------------|----------------------------------------------------------------------------|--------------|----------------------|-------|--------|
| /        | espnet Transformer              | only attention loss, +speed perturb 0.9 1.1, 50 epochs                     | only decoder | /                    | 7.7   | DI WU  |
| /        | opentransformer Transformer     | only attention loss, no speed perturb, 80 epochs                           | only decoder | /                    | 6.7   | Binbin |
| 20201014 | wenet Transformer               | joint CTC, no speed perturb, 20 epoch                                      | only decoder | 20, 1GPU             | 11.71 | Binbin |
| 20201015 | wenet Transformer               | joint CTC, +speed perturb 0.9 1.1, 15 epoch                                | only decoder | 60, 1GPU             | 10.07 | Binbin |
| 20201018 | wenet Transformer               | joint CTC, +speed perturb 0.9 1.1, 20 epoch, fix mask bug                  | only decoder | 60, 1GPU             | 8.13  | Binbin |
| 20201019 | wenet Transformer               | joint CTC, +speed perturb 0.9 1.1, 50 epoch, fix mask bug                  | only decoder | /                    | 6.58  | DI WU  |
| 20201019 | wenet Transformer top5 average  | joint CTC, +speed perturb 0.9 1.1, 50 epoch, fix mask bug                  | only decoder | /                    | 6.32  | DI WU  |
| 20201025 | wenet Transformer top10 average | joint CTC, +speed perturb 0.9 1.1, 80 epoch, fix mask bug                  | only decoder | 20, 4gpu             | 5.93  | DI WU  |
| 20201028 | wenet Transformer top10 average | joint CTC, +speed perturb 0.9 1.1, 80 epoch, fix mask bug, accumulate grad | only decoder | 18, 4gpu             | 5.64  | DI WU  |
| 20201102 | wenet Conformer top10 average   | joint CTC, +speed perturb 0.9 1.1, 80 epoch, fix mask bug, accumulate grad | only decoder | 38, 4gpu             | 4.94  | DI WU  |

