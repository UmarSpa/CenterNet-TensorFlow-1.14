# CenterNet TensorFlow 1.14

This repository contains the implementation of `Centernet` in TensorFlow 1.14

- [CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189), Kaiwen Duan et al. `2019`

## Preparation

Please follow the instructions in the [official repo](https://github.com/Duankaiwen/CenterNet) to install: `NMS`, `MS COCO APIs` and `MS COCO Data`. There is no need to compile corner pooling layers, as this repo contains the code for all the [pooling layers](https://github.com/UmarSpa/CenterNet-TensorFlow-1.14/blob/main/models/CenterNet/pooling.py) directly implemented in TensorFlow 1.14.

  
## Docker

Build the docker image using following command:

```
docker build --tag "cn:tf1.14" --network=host .
```

Once the docker image is build, use following command to run the training:

```
docker run -it --rm --runtime=nvidia --name <your_name>_train --network host -v /data:/data -v <your_workspace_path>/centernet:/code cn:tf1.14 python -u /code/main.py --cfg_file /code/config/config_docker.yaml
```

All the training metrics can be visualized on the tensorboard with the following command:

```
docker run -it --rm --runtime=nvidia --name <your_name>_log --network host -p <your_port>:<your_port> -v <your_workspace_path>/centernet:/code cn:tf1.14 tensorboard --port <your_port> --logdir /code/output/logs
```
