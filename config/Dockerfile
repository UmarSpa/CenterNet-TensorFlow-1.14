FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

ADD huawei_web_secure_internet_gateway_CA.crt /usr/local/share/ca-certificates/huawei_web_secure_internet_gateway_CA.crt
RUN chmod 644 /usr/local/share/ca-certificates/huawei_web_secure_internet_gateway_CA.crt && update-ca-certificates

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update  && apt-get install -y \
    default-jdk \
    python3-tk  \
    ca-certificates \
    protobuf-compiler \
    git

RUN pip install sklearn matplotlib pillow tqdm lxml contextlib2
RUN pip install pyyaml
RUN pip install opencv-python
RUN pip install Cython
RUN pip install scikit-image

RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR cocoapi/PythonAPI
RUN python setup.py build_ext install
RUN rm -rf cocoapi
WORKDIR /tf