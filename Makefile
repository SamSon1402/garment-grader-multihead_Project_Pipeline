.PHONY: install run test demo docker

install:
	python -m pip install -e .[dev]

run:
	uvicorn garment_grader.api.main:app --reload

test:
	pytest -q

demo:
	python scripts/run_demo.py

docker:
	docker build -t garment-grader-industrial .
