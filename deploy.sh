#!/bin/sh
# Creates public vpc + iam roles + security groups + ALB + ECS cluster
aws cloudformation deploy --stack-name=production --template-file=aws/public_vpc.yml --capabilities=CAPABILITY_IAM
# Creates ECS service + ECS task + Link to ALB
aws cloudformation deploy --stack-name=ifpass --template-file=aws/public_service.yml --capabilities=CAPABILITY_IAM
# Prints the service URL
aws cloudformation describe-stacks --stack-name production | grep elb.amazonaws.com
# Bye
exit 0
