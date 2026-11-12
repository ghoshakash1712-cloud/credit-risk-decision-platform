.PHONY: ingest dataset train audit drift simulate all up test

ingest:
	python -m src.pipeline.ingest_data

dataset:
	python -m src.pipeline.build_dataset

train:
	python -m src.pipeline.train
	python -m src.pipeline.make_holdout

audit:
	python -m src.pipeline.fairness

drift:
	python -m src.pipeline.drift

simulate:
	python -m src.pipeline.simulate_shift

all: ingest dataset train audit drift simulate

up:
	docker compose up --build

test:
	pytest -q tests
