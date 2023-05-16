import sagemaker
from sagemaker.pytorch import PyTorch

data_parallel_config = {"pytorchddp":  {"enabled": True}}
model_parallel_config = {"torch_distributed":  {"enabled": True}}

estimator = PyTorch(
    base_job_name="lightening-multinode-test",
    image_uri="929329369485.dkr.ecr.us-west-2.amazonaws.com/ml-sagemaker-testing:normal_base_torch13_py39_0.1",
    role=sagemaker.get_execution_role(),
    sagemaker_session=sagemaker.Session(default_bucket="ml-sagemaker-testing"),
    instance_count=2,
    instance_type="ml.g4dn.12xlarge",
    output_path="s3://ml-sagemaker-testing/multi-node-testing",

    source_dir="./",
    entry_point="train_with_lightning_fabric.py",

    distribution=model_parallel_config,
)

estimator.fit(wait=False)
