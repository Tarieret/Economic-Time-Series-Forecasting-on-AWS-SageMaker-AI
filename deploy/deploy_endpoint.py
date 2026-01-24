import sagemaker
from sagemaker.xgboost.model import XGBoostModel
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Upload the tarball you created earlier to S3
bucket = sagemaker_session.default_bucket()
prefix = "cpi-forecaster-xgboost"
s3_path = sagemaker_session.upload_data("xgboost-model.tar.gz", bucket=bucket, key_prefix=prefix)

# Create the SageMaker Model
xgb_model = XGBoostModel(
    model_data=s3_path,
    role=role,
    entry_point="inference.py",
    framework_version="1.7-1", # Ensure this matches your local xgb version
)

# Deploy to an Endpoint
predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium"
)

print(f"Endpoint deployed: {predictor.endpoint_name}")