# CenterNet

This repository contains our implementation of `Centernet`.

## Relevant papers
- [CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189), Kaiwen Duan et al. `2019` [[code]](https://github.com/Duankaiwen/CenterNet)

## Create docker image
```
sudo docker build --tag "cn:tf1.14" --network=host .
```

## How to run the code:


**Run TRAINING**
```
docker run -it --rm --runtime=nvidia --name <your_name>_train --network host -v /data:/data -v <your_workspace_path>/centernet:/code cn:tf1.14 python -u /code/main.py --cfg_file /code/config/config_docker.yaml
```

**Run TENSORBOARD:**
```
docker run -it --rm --runtime=nvidia --name <your_name>_log --network host -p <your_port>:<your_port> -v <your_workspace_path>/centernet:/code cn:tf1.14 tensorboard --port <your_port> --logdir /code/output/logs
```
