.PHONY: install test generate-data quality features profitability econometrics models validation clv segmentation actions action-simulation optimize montecarlo governance final-report demo-data all app lint

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

validation:
	uv run python -m customer_profitability.models.validation

clv:
	uv run python -m customer_profitability.clv.report

segmentation:
	uv run python -m customer_profitability.segmentation.report

actions:
	uv run python -m customer_profitability.actions.definitions

action-simulation:
	uv run python -m customer_profitability.actions.report

optimize:
	uv run python -m customer_profitability.optimization.report

montecarlo:
	uv run python -m customer_profitability.simulation.report

governance:
	uv run python -m customer_profitability.governance.report

final-report:
	uv run python -m customer_profitability.reporting.final_report

demo-data:
	uv run python -m customer_profitability.utils.demo_sample

all: generate-data quality features profitability econometrics models validation clv segmentation actions action-simulation optimize montecarlo governance final-report demo-data test

app:
	uv run streamlit run app/app.py

lint:
	uv run ruff check src tests app
