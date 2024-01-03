FROM python:3.10

WORKDIR /code

COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

RUN pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install black pylint==2.16.2

# 添加字体
RUN apt-get update && \
    apt-get install -y fonts-texgyre && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Change working directory and run the desired command
COPY ./src/SFC/DRL/maze /code/src/SFC/DRL/maze

WORKDIR /code/src/SFC/DRL/maze
RUN pip install -e .

WORKDIR /code

RUN useradd -lm -u 501 -g 20 ocean

USER ocean

CMD ["/bin/sh", "-c", "sleep infinity"]