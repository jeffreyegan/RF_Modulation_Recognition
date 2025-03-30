FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
LABEL description="RF Basic Modulation Recogniton Container"
LABEL maintainer="Jeffrey Egan"
LABEL version="0.1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install required packages
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    tensorflow[and-cuda] \
    keras \
    scikit-learn \
    jupyter


ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /code

# Set the default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]