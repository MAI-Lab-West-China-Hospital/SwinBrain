#!/bin/bash
docker run --gpus all --rm -ti -v /PATH/TO/SWINBRAIN:/train -v /PATH/TO/CACHE:/cache --ipc=host projectmonai/monai:latest 
