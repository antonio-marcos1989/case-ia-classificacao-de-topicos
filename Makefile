.PHONY: install embeddings build predict test eda

install:
	python3 -m pip install -e .

embeddings:
	PYTHONPATH=src python3 scripts/build_embeddings.py

build:
	PYTHONPATH=src python3 -m classificador build

predict:
	PYTHONPATH=src python3 -m classificador predict

test:
	PYTHONPATH=src python3 -m pytest -q

eda:
	PYTHONPATH=src python3 scripts/eda.py
