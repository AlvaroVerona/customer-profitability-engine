# Action Impact Simulation Report

Every action's effect is simulated *conditional on acceptance* (simulator.py) and blended by acceptance probability into the expected value of *offering* it (incremental_value.py) -- acceptance/effect parameters are documented heuristics (`config.action_simulation`), not fitted models, since no historical offer/acceptance data exists in this project.

## Summary by action

         action_name  n_eligible  mean_acceptance_probability  mean_delta_clv  mean_expected_cost  mean_incremental_profit  total_incremental_profit  pct_beats_no_action
      Credit Product        8170                         0.30         475.046               4.500                  470.546               3844362.564                1.000
  Investment Product        8161                         0.20          26.706               2.000                   24.706                201622.000                1.000
           No Action       12651                         1.00           0.000               0.000                    0.000                     0.000                0.000
Premium Subscription       10921                         0.25          71.260               2.000                   69.260                756392.049                1.000
 Retention Incentive       12651                         0.35          15.161               7.005                    8.156                103183.537                0.700
  Savings Cross-Sell        5791                         0.40          23.850               2.000                   21.850                126530.577                0.826

## Example: recommended actions for customer `CUST_000002`

         action_name  expected_action_cost  delta_clv  incremental_profit
Premium Subscription                  2.00      85.33               83.33
  Investment Product                  2.00      32.17               30.17
  Savings Cross-Sell                  2.00      29.11               27.11
 Retention Incentive                  5.56       8.47                2.91
           No Action                  0.00       0.00                0.00

