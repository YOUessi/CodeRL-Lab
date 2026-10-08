FROM python:3.11-slim

RUN pip install --no-cache-dir numpy==2.2.6

COPY runner.py /runner.py

WORKDIR /work
ENTRYPOINT ["python", "/runner.py"]
