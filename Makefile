.PHONY: install embeddings build predict check test eda all

install:
	python3 -m pip install -e .

embeddings:
	PYTHONPATH=src python3 scripts/build_embeddings.py

build:
	PYTHONPATH=src python3 -m classificador build

predict:
	PYTHONPATH=src python3 -m classificador predict

check:
	PYTHONPATH=src python3 -m classificador check --strict-hash

test:
	PYTHONPATH=src python3 -m pytest -q

eda:
	PYTHONPATH=src python3 scripts/eda.py

# Regeneração completa da entrega + validação (offline, sem LLM)
all: predict check test
