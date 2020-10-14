# Performance Record

| Date     | model       | train                                        | decode       | speed(min/per epoch) | CER   | Add by |
|----------|-------------|----------------------------------------------|--------------|----------------------|-------|--------|
| espnet   | Transformer | joint CTC, +speed perturb 0.9 1.1, 20 epochs | only decoder | /                    | 7.7   | Di Wu  |
| 20201014 | Transformer | joint CTC, no speed perturb, 20 epoch        | only decoder | 20, 1GPU             | 11.71 | Binbin |
| 20201015 | Transformer | joint CTC, +speed perturb 0.9 1.1, 15 epoch  | only decoder | 60, 1GPU             | 10.07 | Binbin |

