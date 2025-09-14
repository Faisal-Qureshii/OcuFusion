# MultiEYE FastAPI Dockerfile
FROM python:3.10.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# system deps for pillow/opencv and fonts for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \# MultiEYE FastAPI Dockerfile
FROM python:3.10.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for pillow/opencv and fonts for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# copy source
COPY . /app

# create runtime dirs
RUN mkdir -p /app/checkpoints /app/analysis

EXPOSE 8000

RUN chmod +x /app/start.sh

CMD ["./start.sh"]

    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# copy source
COPY . /app

# create runtime dirs
RUN mkdir -p /app/checkpoints /app/analysis

# expose port
EXPOSE 8000

# ensure start.sh is executable
RUN chmod +x /app/start.sh

# default command
CMD ["./start.sh"]
