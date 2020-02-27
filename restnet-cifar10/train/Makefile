login:
	aws ecr get-login --no-include-email --region us-east-1 --registry-ids 763104351884
.PHONY: login


build:
	docker build -t resnet-cifar10-train .
.PHONY: build    

