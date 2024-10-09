#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
#RUN apt-get update && apt-get install -y --no-install-recommends \
#	python3-pip \
#	python3-setuptools \
#	build-essential \
#	&& \
#	apt-get clean && \
#	python -m pip install --upgrade pip
#
#WORKDIR /workspace
#COPY ./   /workspace
#
#RUN pip install pip -U
#RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#
#RUN pip install -e .
#
#CMD ["bash", "predict.sh"]

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	python3-pip \
	python3-setuptools \
	build-essential \
	&& \
	apt-get clean && \
	python -m pip install --upgrade pip
WORKDIR /workspace
COPY ./   /workspace

ENV PATH="/root/.local//bin:${PATH}"

RUN python -m pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple --user -U pip
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --user pip-tools
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m piptools sync -i https://pypi.tuna.tsinghua.edu.cn/simple requirements.txt
RUN python -m pip install --user torch-2.0.1+cu117-cp310-cp310-linux_x86_64.whl
RUN pip install -e .

#CMD ["bash", "predict.sh"]
