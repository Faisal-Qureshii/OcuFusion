.PHONY: install run docker-build docker-run clean

install:
	python -m pip install --upgrade pip setuptools wheel
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t multieye-backend:latest .

docker-run:
	docker run --gpus all -p 8000:8000 -v $(shell pwd)/checkpoints:/app/checkpoints -v $(shell pwd)/analysis:/app/analysis --rm -it multieye-backend:latest

clean:
	rm -rf __pycache__ *.pyc .pytest_cache
