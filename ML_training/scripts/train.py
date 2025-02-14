import mlflow
import mlflow.tensorflow

# Start an MLflow run
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)

    # Log metrics during training
    for epoch in range(10):
        loss = 0.01 * epoch  # Example loss value
        mlflow.log_metric("loss", loss, step=epoch)

    # Save and log the model
    model_path = "models/my_model"
    # Save your model here (e.g., TensorFlow or PyTorch save function)
    mlflow.log_artifacts(model_path, artifact_path="model")
