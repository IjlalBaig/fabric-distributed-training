import sagemaker
from sagemaker.pytorch import PyTorch

smp_options = {
    "enabled": True,
    "parameters": {                        # Required
        "partitions": 2,
        "pipeline_parallel_degree": 2,     # Required
        "microbatches": 4,
        "placement_strategy": "spread",
        "pipeline": "interleaved",
        "optimize": "speed",
        "ddp": True,
    }
}

mpi_options = {
    "enabled": True,                      # Required
    "processes_per_host": 8,              # Required
    # "custom_mpi_options" : "--mca btl_vader_single_copy_mechanism none"
}
model_parallel_config = {
        "smdistributed": {"modelparallel": smp_options},
        "mpi": mpi_options
    },

# todo: use config for smdistributed and mpi for ddp case
data_parallel_config = {"pytorchddp":  {"enabled": True}}

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

    # enabling DDP using MPI
    distribution=model_parallel_config,
)

estimator.fit(wait=False)
