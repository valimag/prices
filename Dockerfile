FROM tetsuyahhh/onnxruntime-gpu:latest
RUN rm /etc/apt/sources.list.d/nvidia-ml.list && rm /etc/apt/sources.list.d/cuda.list && apt-get update && apt-get install -y apt-transport-https ffmpeg libsm6 libxext6
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY ./ /app

WORKDIR /app

COPY ./entrypoint.sh /
ENTRYPOINT ["sh", "/entrypoint.sh"]
