from sagemaker.tensorflow import TensorFlowModel
import sagemaker

role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session()

# Path to the tarball created
model_data = f"s3://{sagemaker_session.default_bucket()}/models/lstm-model.tar.gz"

lstm_model = TensorFlowModel(
    model_data=model_data,
    role=role,
    framework_version="2.18", 
    entry_point="inference.py"
)

# Deploy to a small instance
predictor = lstm_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium"
)

print(f"Endpoint deployed: {predictor.endpoint_name}")