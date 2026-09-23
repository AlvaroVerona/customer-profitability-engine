# Optimization Engine Report

Selects the portfolio of customer actions maximizing total incremental economic profit under budget, capacity, risk, and one-action-per-customer constraints (OR-Tools CP-SAT, exact solve). This is an optimal allocation *under Phase 10's simulated action effects* -- a decision-support ranking, not a causal guarantee (PROJECT_SPEC.md 11).

**Solver status: OPTIMAL**

## Constraints

- Budget: 100,000.00 (expected cost basis)
- Operational capacity: 5,000 actions
- Max incremental monthly credit risk: 20,000.00
- Min gross ROI (delta_clv / cost): 0.00

## Outputs

- Candidates considered: 44,406
- Actions selected: 5,000
- Total expected cost: 13,486.23 (remaining budget: 86,513.77)
- Total incremental profit: 1,380,049.68
- Total delta CLV: 1,393,535.82
- Total expected incremental monthly credit risk: 20,000.00
- Expected monthly churns averted (selected retention offers): 0.55

## Action distribution (selected)

         action_type  n_selected
PREMIUM_SUBSCRIPTION        3577
      CREDIT_PRODUCT        1150
  SAVINGS_CROSS_SELL         182
 RETENTION_INCENTIVE          91

## Counterfactual: no optimization vs. optimized allocation

- No optimization (everyone gets NO_ACTION): incremental profit = 0.00, cost = 0.00
- Optimized allocation: incremental profit = 1,380,049.68, cost = 13,486.23
- Value added by running the optimization: 1,380,049.68

## Full recommended action plan: 12,651 active customers, 5,000 receive an intervention

