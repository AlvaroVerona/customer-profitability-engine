"""Expected column dtypes for each raw table.

Used by the generator to cast output consistently and by the Phase 2 data
quality engine to validate structure. Dtypes are pandas dtype strings.
"""

from __future__ import annotations

CUSTOMER_SEGMENTS = ["Mass", "Affluent", "Premium", "Student"]
EMPLOYMENT_STATUSES = ["employed", "self_employed", "unemployed", "student", "retired"]
ACQUISITION_CHANNELS = ["organic", "paid_search", "referral", "partnership", "advertising"]
COUNTRIES = ["ES", "PT", "IT", "FR", "DE", "NL"]
SUPPORT_CHANNELS = ["phone", "chat", "email", "in_app"]
PRODUCTS = [
    "current_account",
    "savings_account",
    "debit_card",
    "credit_card",
    "consumer_loan",
    "investment_account",
]

CUSTOMERS_SCHEMA: dict[str, str] = {
    "customer_id": "string",
    "age": "int16",
    "income": "float64",
    "country": "category",
    "acquisition_channel": "category",
    "acquisition_date": "datetime64[ns]",
    "customer_segment": "category",
    "employment_status": "category",
    "rate_sensitivity": "float64",
    "satisfaction_baseline": "float64",
    "churn_flag": "boolean",
    "churn_date": "datetime64[ns]",
}

ACCOUNTS_SCHEMA: dict[str, str] = {
    "customer_id": "string",
    "product": "category",
    "account_open_date": "datetime64[ns]",
    "account_close_date": "datetime64[ns]",
    "is_active": "boolean",
}

DEPOSITS_SCHEMA: dict[str, str] = {
    "customer_id": "string",
    "month": "datetime64[ns]",
    "product": "category",
    "opening_balance": "float64",
    "closing_balance": "float64",
    "average_balance": "float64",
    "inflows": "float64",
    "outflows": "float64",
    "deposit_rate": "float64",
    "market_rate": "float64",
}

CARDS_SCHEMA: dict[str, str] = {
    "customer_id": "string",
    "month": "datetime64[ns]",
    "transaction_count": "int32",
    "transaction_volume": "float64",
    "interchange_revenue": "float64",
    "atm_withdrawal_volume": "float64",
    "international_transaction_volume": "float64",
    "payment_processing_cost": "float64",
}

LOANS_SCHEMA: dict[str, str] = {
    "loan_id": "string",
    "customer_id": "string",
    "month": "datetime64[ns]",
    "origination_date": "datetime64[ns]",
    "outstanding_balance": "float64",
    "credit_limit": "float64",
    "utilization": "float64",
    "interest_rate": "float64",
    "interest_income": "float64",
    "pd": "float64",
    "lgd": "float64",
    "ead": "float64",
}

CUSTOMER_SERVICE_SCHEMA: dict[str, str] = {
    "customer_id": "string",
    "month": "datetime64[ns]",
    "support_contacts": "int16",
    "average_handling_time": "float64",
    "support_channel": "category",
    "estimated_service_cost": "float64",
}

CUSTOMER_MONTHLY_SCHEMA: dict[str, str] = {
    "customer_id": "string",
    "month": "datetime64[ns]",
    "tenure_months": "int32",
    "is_active": "boolean",
    "product_count": "int8",
    "total_deposit_balance": "float64",
    "total_loan_balance": "float64",
    "transaction_count": "int32",
    "transaction_volume": "float64",
    "support_contacts": "int16",
    "login_frequency": "float64",
    "satisfaction_proxy": "float64",
    "churned_this_month": "boolean",
}

TABLE_SCHEMAS: dict[str, dict[str, str]] = {
    "customers": CUSTOMERS_SCHEMA,
    "accounts": ACCOUNTS_SCHEMA,
    "deposits": DEPOSITS_SCHEMA,
    "cards": CARDS_SCHEMA,
    "loans": LOANS_SCHEMA,
    "customer_service": CUSTOMER_SERVICE_SCHEMA,
    "customer_monthly": CUSTOMER_MONTHLY_SCHEMA,
}

# Columns whose combination should uniquely identify a row. Phase 2 uses this
# both to detect duplicate business records and to flag rows with a missing
# key component (which breaks joins and must never be silently dropped).
PRIMARY_KEYS: dict[str, list[str]] = {
    "customers": ["customer_id"],
    "accounts": ["customer_id", "product"],
    "deposits": ["customer_id", "month", "product"],
    "cards": ["customer_id", "month"],
    "loans": ["loan_id", "month"],
    "customer_service": ["customer_id", "month"],
    "customer_monthly": ["customer_id", "month"],
}

# Missing-value severity per (table, column), used by the Phase 2 quality
# engine to distinguish expected / suspicious / critical missingness rather
# than treating all nulls the same way.
#   - critical:   breaks identity, joins, or a required financial calculation
#                 -> the row is quarantined.
#   - suspicious: degrades a calculation but doesn't invalidate the row
#                 -> the row stays validated, but is counted and reported.
#   - expected:   a soft/optional field that is legitimately sometimes absent.
MISSING_SEVERITY: dict[str, dict[str, str]] = {
    "customers": {"churn_date": "expected"},  # NaT simply means "still active"
    "accounts": {"account_close_date": "expected"},  # NaT simply means "still open"
    "deposits": {"average_balance": "suspicious", "deposit_rate": "suspicious"},
    "cards": {"transaction_volume": "suspicious"},
    "customer_service": {"average_handling_time": "expected"},
    "customer_monthly": {"satisfaction_proxy": "expected"},
    "loans": {"interest_rate": "critical"},
}
