# IFPaaS - Iris Flower Prediction as a Service [![Build Status](https://travis-ci.com/monolli/IFPaaS.svg?branch=master)](https://travis-ci.com/monolli/IFPaaS)

This repository contains the implementation of machine learning microservice and its deployment to the cloud as a container. The objective is to provide a cloud hosted API capable of predicting the class of an Iris flower sample.

<p style="text-align:center;"><img src="/imgs/api.png" alt="The API" width="90%"></p>

## Project overview

### The data

The chosen dataset was the famous "Fisher\`s Iris Flower Data", it is publicly available and can be downloaded from a lot of trusted sources. In this project the data is imported from the Scikit-learn's Python package.

The following sample represents all the available classes/species and features of the original dataset.

| sepal_length | sepal_width | petal_length | petal_width | species    |
| ------------ | ----------- | ------------ | ----------- | ---------- |
| 5.1          | 3.5         | 1.4          | 0.2         | setosa     |
| 7.0          | 3.2         | 4.7          | 1.4         | versicolor |
| 6.3          | 3.3         | 6.0          | 2.5         | virginica  |


### The model

The model uses the Random Forest algorithm from the Scikit-learn package in order to predict the class of the flower. The trained model is serialized and saved as a Pickle object.

A separated module was developed for the prediction step. It loads the Pickle object and receives the dataset containing the objects that are going to be classified, after that, it returns the predicted species of each received object.

### The API

The API was developed using Flask for the web app definition, RESTPlus (Swagger) for the interactive documentation page, and Gunicorn as the WSGI.

### The CI/CD process

The pipeline was built using Travis CI, the CI/CD process uses Pytest as the test suit, Flake8 for linting, Docker for containerization, and DockerHub as the container repository. By the end of the pipeline the app is deployed to the AWS, and in order to run the app in the cloud, the infrastructure is generated using CloudFormation (infrastructure as code).

<p style="text-align:center;"><img src="/imgs/pipeline_diagram.png" alt="The pipeline diagram" width="80%"></p>

### The AWS architecture

The proposed infrastructure uses an isolated public VPC with an attached Internet Gateway, an Application Load Balancer (ALB) for scalability, and ECS + Fargate to host the Docker container. By using Fargate the container is hosted as a service, in other words, no Docker/Kubernetes cluster cost and management overhead.

<p style="text-align:center;"><img src="/imgs/architecture.png" alt="The aws architecture" width="85%"></p>

## Usage

### Prerequisites

You must have Docker installed.

### Running

The container is hosted as DockerHub, so just download it:

```sh
docker pull monolli/ifpass:latest
```
Run it using the following command:
```sh
docker run -p 8000:8000 --rm -it monolli/ifpass
```
**It should now be available at http://0.0.0.0:8000.**

You can send requests to the API using the "Try it out" button inside the documentation, or by sending your custom requests as the following example:
```sh
curl -X POST "http://0.0.0.0:8000/classify_iris/multiple" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"data\": [[4.8,3.4,1.6,0.2],[6.5,3.2,5.1,2]]}"
```

## Author

* **Lucas Monteiro de Oliveira**- [Monolli](https://github.com/monolli)

## License

This project is licensed under the GNU GPL3 License - see the [LICENSE.md](LICENSE.md) file for details
