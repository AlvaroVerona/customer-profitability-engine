"""Phase 5.3 -- Econometric diagnostics shared by the revenue and churn models.

Provides model-agnostic helpers (statsmodels `OLS` and `Logit` results
objects expose the same `.params` / `.bse` / `.pvalues` / `.tvalues` /
`.conf_int()` interface) for:

- a coefficient table with standard errors, test statistics, p-values, and
  95% confidence intervals
- multicollinearity (VIF)
- residual normality (Jarque-Bera)
- heteroskedasticity (Breusch-Pagan) -- OLS only, meaningless for Logit
- goodness of fit (R^2/adjusted R^2 for OLS, McFadden pseudo-R^2 for Logit)

Development Principle #13 ("do not treat statistical significance as
economic significance") is implemented directly:
`flag_economic_significance` requires the caller to state, per coefficient,
what a *practically meaningful* effect size would be -- a coefficient can be
statistically significant (p < 0.05) with a trivial effect, or
non-significant with too little power to rule out a large one, and the two
questions are kept in separate columns rather than collapsed into one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera


def coefficient_table(results) -> pd.DataFrame:
    ci = results.conf_int()
    ci.columns = ["ci_lower", "ci_upper"]
    return pd.DataFrame(
        {
            "coef": results.params,
            "std_err": results.bse,
            "statistic": results.tvalues,
            "p_value": results.pvalues,
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "significant_p05": results.pvalues < 0.05,
        }
    )


def flag_economic_significance(coef_table: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """`thresholds[var]` is the smallest |coefficient| the analyst considers
    practically meaningful for that variable's units. A row can be
    statistically significant but economically negligible, or vice versa --
    both are reported, not merged into a single verdict."""
    out = coef_table.copy()
    out["economic_threshold"] = out.index.map(thresholds).fillna(np.inf)
    out["economically_significant"] = out["coef"].abs() >= out["economic_threshold"]
    return out


def compute_vif(exog: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor per column of `exog` (must already include
    the constant, if any -- VIF on the constant itself is not meaningful and
    is dropped from the result)."""
    vifs = {
        col: variance_inflation_factor(exog.to_numpy(), i)
        for i, col in enumerate(exog.columns)
        if col != "const"
    }
    return pd.Series(vifs, name="vif")


def residual_normality(results) -> dict[str, float]:
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(results.resid)
    return {
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
    }


def breusch_pagan_test(results) -> dict[str, float]:
    """Tests whether OLS residual variance depends on the regressors
    (heteroskedasticity). Not applicable to Logit models."""
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(results.resid, results.model.exog)
    return {
        "lm_statistic": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pvalue),
        "heteroskedastic_p05": bool(lm_pvalue < 0.05),
    }


def goodness_of_fit_ols(results) -> dict[str, float]:
    return {
        "r_squared": float(results.rsquared),
        "adj_r_squared": float(results.rsquared_adj),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "n_obs": int(results.nobs),
    }


def goodness_of_fit_logit(results) -> dict[str, float]:
    """McFadden's pseudo-R^2 = 1 - (log-likelihood of full model) /
    (log-likelihood of intercept-only model)."""
    return {
        "pseudo_r_squared_mcfadden": float(results.prsquared),
        "log_likelihood": float(results.llf),
        "log_likelihood_null": float(results.llnull),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "n_obs": int(results.nobs),
    }


def add_constant(exog: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(exog, has_constant="add")
