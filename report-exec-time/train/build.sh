#!/usr/bin/env bash

image=$1

region=$(aws configure get region)

echo $region

$(aws ecr get-login --region ${region} --no-include-email)

docker build -t ${image} .

