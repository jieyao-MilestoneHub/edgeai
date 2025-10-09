"""MLflow 模型註冊"""
import mlflow
import mlflow.pytorch

def register_model(model, model_name: str, params: dict):
    """註冊模型到 MLflow"""
    with mlflow.start_run():
        # 記錄參數
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # 記錄模型
        mlflow.pytorch.log_model(model, "model")
        
        # 註冊到 registry
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            model_name
        )
