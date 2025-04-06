import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

class ShoppingPredictorModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        # Encode categorical features
        categorical_features = ['Month', 'VisitorType', 'Weekend']
        encoded_cats = pd.DataFrame(
            self.encoder.fit_transform(data[categorical_features]),
            columns=self.encoder.get_feature_names_out(categorical_features)
        )

        # Combine with numerical features
        numerical_features = data.drop(columns=categorical_features + ['Revenue'])
        processed_data = pd.concat([numerical_features, encoded_cats], axis=1)

        # Scale data
        scaled_data = self.scaler.fit_transform(processed_data)
        return scaled_data, data['Revenue']

    def train_model(self, data_path):
        data = self.load_data(data_path)
        X, y = self.preprocess_data(data)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train logistic regression model
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        # Evaluate the model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Save the model
        joblib.dump(self.model, 'purchase_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.encoder, 'encoder.pkl')

    def predict(self, input_data):
        # Load the trained model
        if self.model is None:
            self.model = joblib.load('purchase_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.encoder = joblib.load('encoder.pkl')

        # Convert input data to dataframe
        input_df = pd.DataFrame([input_data])

        # Encode and scale the data
        encoded_input = pd.DataFrame(
            self.encoder.transform(input_df[['Month', 'VisitorType', 'Weekend']]),
            columns=self.encoder.get_feature_names_out(['Month', 'VisitorType', 'Weekend'])
        )
        input_data_combined = pd.concat([input_df.drop(columns=['Month', 'VisitorType', 'Weekend']), encoded_input], axis=1)
        scaled_input = self.scaler.transform(input_data_combined)

        # Predict the result
        prediction = self.model.predict(scaled_input)[0]
        return "Will Purchase" if prediction == 1 else "Will Not Purchase"
