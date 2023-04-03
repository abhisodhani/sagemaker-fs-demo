# sagemaker-fs-demo

Overview:

In this repository, we provide artifacts that demonstrate how to leverage Amazon SageMaker Feature Store and Kinesis Data Analytics for streaming feature aggregation. Our use case is Fraud Detection on credit card transactions. We use Amazon SageMaker to train a model (using the built-in XGBoost algorithm) with aggregate features created from historical credit card transactions. We use streaming aggregation with Amazon Kinesis and Amazon Kinesis Data Analytics (KDA) SQL, publishing features in near real time to SageMaker Feature Store. Finally, we pull the latest aggregate feature values from the feature store at inference time, passing them as input to our fraud detection model hosted in an Amazon SageMaker endpoint.