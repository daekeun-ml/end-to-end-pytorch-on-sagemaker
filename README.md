# Building an end-to-end ML Image Classification demo based on the PyTorch framework on Amazon SageMaker

## Introduction

### Purpose
- MNIST, CIFAR-10와 같은 toy-data가 아닌 real-world에 좀 더 가까운 데이터에 대한 AIML 학습/배포 모범 사례 니즈가 강함
- Amazon SageMaker 상에서의 Pytorch End-to-end AIML 예제 코드에 대한 니즈 
- PyTorch 1.5.0부터 지원되는 apex 기반 분산학습 수행, torchserving으로 엔드포인트(Endpoint)를 배포하는 모범 사례 구축 니즈

### Scenario
- Training Data
    - Bengali.AI Handwritten Grapheme Classification (https://www.kaggle.com/c/bengaliai-cv19)
    - MNIST와 달리 168개 클래스+11개 클래스+7개 클래스를 동시예 예측해야 하는 복잡한 문제임
    - Evaluation metric: Weighted Recall (각 클래스에 대한 2:1:1 비중), baseline 약 90%, 1위 97%
- Preprocessing
    - parquet → direct training data input에 대한 research (daekeun@)
- Training
    - 기본 학습 코드 작성, CutMix 적용 (daekeun@)
    - SageMaker BYOS + apex 기반 분산 학습 코드 작성 (choijoon@)
    - Optional: Self-Supervised Learning
- Deployment
    - MMS, Elastic Inference (daekeun@)
    - torchserving (daekeun@)

### Contributors
- Daekeun Kim (daekeun@amazon.com)
- Youngjoon Choi (choijoon@amazon.com)

## Implementation

### [Module 1. EDA and Data Preparation](1.eda_and_data_split.ipynb)

### [Module 2. Training without Amazon SageMaker](2.training_local.ipynb)

### [Module 3. Training on Amazon SageMaker](3.training_on_sagemaker.ipynb)

### [Module 3b. (Optional) Training on Amazon SageMaker using SageMaker Data Parallelism Library](3b.training_on_sagemaker_smdataparallel.ipynb)

### [Module 4. Multi-GPU Distributed Training on Amazon SageMaker](4.distributed_training_on_sagemaker.ipynb)

### [Module 5. Deployment on MMS(Multi Model Server)](5.deployment.ipynb)

### [Module 6. Deployment for EIA(Elastic Inference Accelerator)](6.deployment_eia.ipynb)


## CloudFormation
본 핸즈온랩에 필요한 AWS 리소스를 생성하기 위해 CloudFormation 스택이 제공됩니다. 아래 링크를 선택하면 스택이 시작될 AWS 콘솔의 CloudFormation 으로 자동 redirection 됩니다.

- [Launch CloudFormation stack in ap-northeast-2 (Seoul)](https://console.aws.amazon.com/cloudformation/home?region=ap-northeast-2#/stacks/create/review?stackName=AIMLWorkshop&templateURL=https://daekeun-workshop-public-material.s3.ap-northeast-2.amazonaws.com/cloudformation/sagemaker-imd.yaml)
- [Launch CloudFormation stack in us-east-1 (N. Virginia)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?stackName=AIMLWorkshop&templateURL=https://daekeun-workshop-public-material.s3.ap-northeast-2.amazonaws.com/cloudformation/sagemaker-imd.yaml)
- [Launch CloudFormation stack in us-east-2 (Ohio)](https://console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/create/review?stackName=AIMLWorkshop&templateURL=https://daekeun-workshop-public-material.s3.ap-northeast-2.amazonaws.com/cloudformation/sagemaker-imd.yaml)
- [Launch CloudFormation stack in us-west-1 (N. California)](https://console.aws.amazon.com/cloudformation/home?region=us-west-1#/stacks/create/review?stackName=AIMLWorkshop&templateURL=https://daekeun-workshop-public-material.s3.ap-northeast-2.amazonaws.com/cloudformation/sagemaker-imd.yaml)
- [Launch CloudFormation stack in us-west-2 (Oregon)](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?stackName=AIMLWorkshop&templateURL=https://daekeun-workshop-public-material.s3.ap-northeast-2.amazonaws.com/cloudformation/sagemaker-imd.yaml)

CloudFormation 스택은 아래 리소스를 자동으로 생성합니다.

- EC2 및 SageMaker 인스턴스에 퍼블릭 서브넷 + 보안 그룹이 있는 VPC
- AWS 리소스에 액세스하는 데 필요한 IAM role
- Jupyter 노트북에서 모델을 정의하는 SageMaker 노트북 인스턴스. 모델 자체는 SageMaker 서비스를 사용하여 학습됩니다.
- SageMaker에 필요한 S3 버킷

AWS CloudFormation 콘솔의 Quick create stack 페이지로 이동 후 다음 단계를 수행하여 스택을 시작합니다.

- DefaultCodeRepository: 발표자가 안내하는 GitHub 리포지토리 주소를 지정합니다. 
(예: https://github.com/daekeun-ml/end-to-end-pytorch-on-sagemaker)
- MLInstanceType: SageMaker notebook instance type을 선택합니다. 미리 설정된 **`ml.m4.xlarge`** 나 **`ml.t2.medium`** 인스턴스를 권장합니다. 
- Capabilities: **`I acknowledge that AWS CloudFormation might create IAM resources`** 을 체크합니다.
- 우측 하단의 **`Create stack`** 버튼을 누르고, 스택 생성이 완료될 때까지 기다립니다. 약 10분이 소요됩니다

<br>

[Privacy](https://aws.amazon.com/privacy/) | [Site terms](https://aws.amazon.com/terms/) | © 2021, Amazon Web Services, Inc. or its affiliates. All rights reserved.