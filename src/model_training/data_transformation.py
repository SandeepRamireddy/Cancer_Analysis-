from sklearn.preprocessing import StandardScaler
import pickle
import os
from src.model_training.data_initialization import split_data


class DataTransformer:
    def __init__(self):
        self.scaler = StandardScaler()

    def transform_and_save(self):
        """
        Split data, encode labels, scale, save scaler, return train/test splits.
        """
        # Split
        x_train, x_test = split_data("Cancer_Data.csv")

        # Encode target
        x_train["diagnosis"] = [1 if i == "M" else 0 for i in x_train["diagnosis"]]
        y_train = x_train["diagnosis"]
        x_train_data = x_train.drop('diagnosis',axis =1)
        x_test["diagnosis"] = [1 if i == "M" else 0 for i in x_test["diagnosis"]]
        y_test = x_test["diagnosis"]
        x_test_data = x_test.drop('diagnosis',axis =1)
        

        # Scale
        x_train_scaled = self.scaler.fit_transform(x_train_data)
        x_test_scaled = self.scaler.transform(x_test_data)

        # Save scaler in artifacts
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        scaler_path = os.path.join(base_dir, "artifacts", "scaler.pkl")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Scaler saved at {scaler_path}")
        return x_train_scaled, x_test_scaled, y_train, y_test


if __name__ == "__main__":
    transformer = DataTransformer()
    x_train, x_test, y_train, y_test = transformer.transform_and_save()


