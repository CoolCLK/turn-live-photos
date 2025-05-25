FROM python:3.10.6

RUN useradd -m -u 1000 user

WORKDIR /app

COPY . .

RUN pip install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt

USER user
RUN python __main__.py