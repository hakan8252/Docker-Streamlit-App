
Telco Churn Prediction Web App
Welcome to the Telco Churn Prediction Web App! This web application predicts Telco churn values using a LightGBM classifier model.

Prerequisites
Before running the app, ensure you have the following installed:

Docker to build and run the Docker container.
Basic knowledge of Docker commands and concepts.
Getting Started
Follow these steps to run the Telco Churn Prediction Web App locally:

1. Clone the Repository
```bash
git clone https://github.com/hakan8252/Docker-Streamlit-App.git
```

2. Pull the Docker image from the repository:
```bash
docker pull hakan8252/streamlit-app:version1
docker run -p 8501:8501 hakan8252/streamlit-app:version1
```

4. Access the Web App
Once the container is running, you can access the web app by opening your browser and navigating to http://localhost:8501.

About the App
The Telco Churn Prediction Web App is built using Streamlit, a popular Python library for creating web apps with minimal code. The app leverages a LightGBM classifier model trained on Telco customer data to predict churn values.

Usage
Upon accessing the web app, you'll be presented with an interface to input customer data.
Just click the "Show Example" button to get the churn prediction for the customer.
The app will display the predicted churn value along with the probability.

Model Details
The Telco churn prediction model is trained using the LightGBM algorithm, a gradient boosting framework. It is optimized for speed, accuracy, and handling large datasets efficiently.

Contributing
If you encounter any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

Project Link: https://docker-churn-prediction-app.streamlit.app/
