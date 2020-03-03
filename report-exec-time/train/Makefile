login-direct:
	$(aws ecr get-login --region ${region} --no-include-email)
.PHONY: login-direct

login:
	aws ecr get-login --no-include-email --region us-east-1 --registry-ids 763104351884
.PHONY: login

build:
	docker build -t report-exec-time-train .
.PHONY: build

build-sh:
	sh build.sh report-exec-time-train
.PHONY: build-sh

