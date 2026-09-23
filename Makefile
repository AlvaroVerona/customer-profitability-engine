.PHONY: install test generate-data quality features profitability econometrics models clv segmentation actions action-simulation optimize app lint

install:
	uv sync --extra dev

test:
	uv run python -m pytest -v

generate-data:
	uv run python -m customer_profitability.data.generator

quality:
	uv run python -m customer_profitability.data.quality


features:
	uv run python -m customer_profitability.features.customer_features

profitability:
	uv run python -m customer_profitability.profitability.engine

econometrics:
	uv run python -m customer_profitability.econometrics.report

models:
	uv run python -m customer_profitability.models.report

clv:
	uv run python -m customer_profitability.clv.report

segmentation:
	uv run python -m customer_profitability.segmentation.report

actions:
	uv run python -m customer_profitability.actions.definitions

action-simulation:
	uv run python -m customer_profitability.actions.report

optimize:
	uv run python -c "print('Phase 11 not implemented yet')"

app:
	uv run streamlit run app/app.py

lint:
	uv run ruff check src tests
