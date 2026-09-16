ARG BASE_IMAGE=docker.1ms.run/python:3.11-slim | mirror.baidubce.com/library/python:3.11-slim

FROM ${BASE_IMAGE}

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY data/knowledge ./data/knowledge

EXPOSE 8000 8001 8002 9001 9002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
