FROM nvcr.io/nvidia/pytorch:24.06-py3
WORKDIR /workspace/app
COPY requirements.txt .
RUN pip install -r requirements.txt