import logging

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    random_state = 42
    housing = fetch_california_housing(as_frame=True)

    X = housing.data
    y = housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    params = {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.15 ,"random_state": random_state}
    run_name = "Production-model"
    model_name = "Production-model"
    explainer_name = "explainer"
    with mlflow.start_run(run_name=run_name):
        mlflow.sklearn.autolog(registered_model_name=model_name)
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        logger.info("Model training completed.")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        eval_data = X_test.copy()
        eval_data["target"] = y_test

        result = mlflow.evaluate(
            model=model_uri,  
            data=eval_data,
            targets="target",
            model_type="regressor",
            evaluators="default",
            evaluator_config={
                "log_explainer": True,
                "explainer_artifact_path": explainer_name,
                "explainer_type": "permutation"
            },
        )
        logger.info(f"Evaluation completed. Artifacts: {result.artifacts}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

if __name__ == "__main__":
    main()
