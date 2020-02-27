build:
	docker build -t saved_model_half_plus_three_serving .
.PHONY: build

local-run:
	docker run --rm -p 8080:8080 saved_model_half_plus_three_serving
.PHONY: local-run



local-query:
	curl -X POST http://localhost:8080/invocations -d '{"instances": [1.0,2.0,5.0]}'
.PHONY: local-query

# Result:
# {
#     "predictions": [3.5, 4.0, 5.5]
# }


local-query2:
	curl -X POST http://localhost:8080/invocations -d '{"instances": [1.5,3.5,4.0]}'
.PHONY: local-query2


local-query3:
	curl -X POST http://localhost:8080/invocations -d '{"instances": [2.0,2.5,3.5]}'
.PHONY: local-query3


push:
	sh push.sh
.PHONY: push

create-model:
	sh sagemaker_model.sh
.PHONY: sagemaker_model

create-endpoint:
	sh sagemaker_endpoint.sh
.PHONY: sagemaker_endpoint

endpoint-status:
	aws sagemaker describe-endpoint --endpoint-name half-plus-three
.PHONE: endpoint-status

