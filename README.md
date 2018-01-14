## What is this project?
This project is my solution of [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) competition hosted by [Kaggle](https://www.kaggle.com) My solution includes using a variety of different approaches including KNN, SVM, DNN RRF, GB.

## What are the dependencies for this project?
* sklearn(for knn,svm,rrf,gbm models)
* Keras(for keras dnn model)
* PyTorch(for PyTorch dnn model)

## Results of each model on Kaggle's leaderboard:
* GB+RRF score: <b>0.36014 (13th place out of 3251)</b></br>
* GB score: 0.36241</br>
* RRF score: 0.37429</br> 
* DNN score: 0.41125(keras code)</br>
* SVR score: 0.44863 (gaussian kernel) </br>
* KNN score: 0.55171 (k=4) </br></br>

![alt text](https://github.com/PiotrSobczak/BikeSharingDemand-Kaggle-ML-competition/blob/master/plots/result_comparison.png)

## Example showing accuracy of the best model:
![alt text](https://github.com/PiotrSobczak/BikeSharingDemand-Kaggle-ML-competition/blob/master/plots/predictions_72h.png)

## Data exploration:
* hour impact on count in working and non-working days
![alt text](https://github.com/PiotrSobczak/BikeSharingDemand-Kaggle-ML-competition/blob/master/plots/hour.png)
* month and season impact on count
![alt text](https://github.com/PiotrSobczak/BikeSharingDemand-Kaggle-ML-competition/blob/master/plots/month_season.png)
* year impact on count
![alt text](https://github.com/PiotrSobczak/BikeSharingDemand-Kaggle-ML-competition/blob/master/plots/year_month.png)
* temperature impact on count
![alt text](https://github.com/PiotrSobczak/BikeSharingDemand-Kaggle-ML-competition/blob/master/plots/temp_atemp.png)
* humidity and wind speed impact on count
![alt text](https://github.com/PiotrSobczak/BikeSharingDemand-Kaggle-ML-competition/blob/master/plots/humidity_wind_speed.png)
