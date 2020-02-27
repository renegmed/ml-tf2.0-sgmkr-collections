login:
	aws ecr get-login --no-include-email --region us-east-1 --registry-ids 763104351884
.PHONY: login

build:
	docker build -t fashion-mnist-train .	
.PHONY: build

#local-run:
#	docker run --rm -p 8080:8080 fashion-mnist-serving
#.PHONY: local-run

#local-query:
#	curl -X POST http://localhost:8080/invocations -d '{"instances": [[  0,   0,   0,   0,   0, 182, 194, 189, 218, 219, 222, 229, 208, 194, 200, 192, 158, 168,  229, 199, 178, 168,  95,   0,   0,   0,   0,   0]]}'
#.PHONY: local-query

