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
