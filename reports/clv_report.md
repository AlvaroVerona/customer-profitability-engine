# Customer Lifetime Value Report

`historical_economic_profit` and `future_clv` are kept separate (PROJECT_SPEC.md 7.4); `total_customer_economic_value` sums them. `clv_p5`/`clv_p50`/`clv_p95` are a Monte Carlo uncertainty band (1,000 simulations/customer) around `total_customer_economic_value` -- CLV is not presented as a single exact number (7.5).

- Customers: 20000 total, 12651 still active as of the observation window end
- Total historical economic profit: 11,291,852
- Total future CLV (active customers only): 10,486,500
- Total customer economic value: 21,778,352
- Mean total CEV (active): 1,596.31
- Median total CEV, Monte Carlo (active): 1,552.12

## Distribution (active customers)

       historical_economic_profit  future_clv  total_customer_economic_value    clv_p5   clv_p50   clv_p95
count                    12651.00    12651.00                       12651.00  12651.00  12651.00  12651.00
mean                       767.41      828.91                        1596.31    971.31   1552.12   2328.68
std                       1065.53     1023.87                        1989.53   1421.25   1989.75   2689.88
min                       -125.27     -523.23                        -347.75  -1864.98   -191.88   -129.26
5%                           9.02        6.43                          32.72      4.95     21.61     85.03
25%                        127.68      189.46                         391.94    160.99    335.19    715.74
50%                        404.08      485.10                         945.78    497.02    900.00   1444.51
75%                        942.92     1050.80                        1981.76   1181.29   1933.77   2815.63
95%                       2899.65     2958.54                        5657.86   3709.81   5596.03   7851.76
max                      14244.86    11401.42                       25646.27  19307.62  25788.12  29099.07

