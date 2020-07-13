# Building an end-to-end ML demo based on the PyTorch framework on Amazon SageMaker

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
    - 기본 학습 코드 작성 (daekeun@)
    - SageMaker BYOS + apex 기반 분산 학습 코드 작성 (choijoon@)
    - Optional: Self-Supervised Learning, CutMix 적용
- Deployment
    - torchserving (daekeun@, choijoon@)
    - eia (choijoon@)

### Contributor
- Daekeun Kim (daekeun@amazon.com)
- Youngjoon Choi (choijoon@amazon.com)

## Implementation (To be updated soon)

### [Module 1. EDA and Data Preparation](module1-eda_and_data_split.ipynb)

### [Module 2. Training without Amazon SageMaker](module2-training_with_cutmix.ipynb)
