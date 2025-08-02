# Health_Insurance_Offer_Response_Prediction

This project aims to predict whether a customer will accept a health insurance offer based on demographic and behavioral features. The dataset used for this project comes from an insurance company and includes features such as age, vehicle damage history, annual premium, and more.

Project Goals

	•	Predict customer response to a health insurance offer (Response: 1 = accepted, 0 = not accepted)
	•	Perform exploratory data analysis to uncover trends and patterns
	•	Train a machine learning model (Random Forest)
	•	Evaluate performance with classification metrics and confusion matrix
	•	Identify the most influential features
	•	Provide business insights based on results
 
⸻⸻

 Dataset Overview
 
	•	Rows: 381,109
	•	Target Variable: Response
	•	Important Features: Age, Vehicle_Damage, Annual_Premium, Vintage
	•	Data Source: Kaggle – Health Insurance Cross Sell Prediction
 
⸻⸻

Exploratory Data Analysis (EDA)

Class Distribution:

	 Majority of customers declined the insurance offer.

Age vs Response:

	Older individuals are more likely to accept the offer.

Vehicle Damage Effect:

	Customers with a history of vehicle damage tend to respond more positively.

<p align="center">
  <img src="Health Insurance/visuals/class_distribution.png" width="400"/>  
  <img src="Health Insurance/visuals/ageresponse_boxplot.png" width="400"/>
</p>

⸻⸻

Model

	•	Model Used: RandomForestClassifier
	•	Scaler: StandardScaler
	•	Train-Test Split: 80/20
	•	Evaluation Metrics: Accuracy, Precision, Recall, F1-score
 
⸻⸻

Classification Report (Sample):

	Metric		Class 0	Class 1
	Precision	0.89	0.37
	Recall		0.97	0.12
	F1-score	0.93	0.18

	The model performs well on the majority class but struggles with recall on the minority class due to dataset imbalance.

<p align="center">
  <img src="Health Insurance/visuals/confusion_matrix.png" width="400"/>
</p>

⸻⸻

Feature Importance

Top features contributing to model decisions:

	•	Vintage (time with company)
	•	Annual_Premium
	•	Age
	•	Vehicle_Damage

<p align="center">
  <img src="Health Insurance/visuals/feature_importance.png" width="400"/>
</p>
