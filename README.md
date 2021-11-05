# Neural_Network_Charity_Analysis

# Overview of Credit Risk Analysis

### The Analysis of credit risk involves unbalanced classifications. Low risk customers outnumber the high risk loans. The imbalanced-learn and scikit-learn libraries will be used to build models and evaluate them with resampling to predict credit risk.
### The credit card credit dataset from LendingClub will be analyzed with two oversampling and one undersampling algorithms and the results from machine learning models will be compared. Next two ensemble models will be used and the performance evaluated. All the results and the summary will focus on the high risk outcome since that is what is of interest in credit risk analyses.

<br/>

## Results: Deliverable 1: Use Resampling Models to Predict Credit Risk 

<br/>

-   RandomOverSampler 

#### The accuracy not great with the Random Over Sampler model (65%). Only 70 of the 101 high risk category are correctly classified (Recall = 69%). The precision (1%) is very low due to a very imbalanced data set with such high number of low risk clients. The F1 score is really low as well (0.02) which indicates a model that is not great for predicting high risk loans.
    
-       Accuracy score
    ![ROS accuracy](./Resources/ros_accuracy.png) 

-       Confusion matrix
    ![ROS matrix](./Resources/ros_matrix.png) 

-       Imbalanced classification report
    ![ROS report](./Resources/ros_report.png) 

<br/>
<br/>

-   SMOTE 

#### The accuracy not great with the Synthetic Minority Oversampling Technique (SMOTE) model (66%). Only 64 of the 101 high risk category are correctly classified (Recall = 63%). The precision (1%) is very low due to a very imbalanced data set with such high number of low risk clients. The F1 score is really low as well (0.02) which indicates a model that is not great for predicting high risk loans. This model is worse that the Random Over Sampler.

       Accuracy score
  
  ![SMO accuracy](./Resources/smo_accuracy.png) 

-       Confusion matrix
    ![SMOTE matrix](./Resources/smo_matrix.png) 

-       Imbalanced classification report
    ![SMOTE report](./Resources/smo_report.png) 

<br/>
<br/>

-   ClusterCentroids

#### The accuracy even lower with the Cluster Centroid model (54%). Only 70 of the 101 high risk category are correctly classified (Recall = 69%). Again the precision (1%) is very low due to a very imbalanced data set with such high number of low risk clients. The F1 score is really low as well (0.01) which indicates a model that is not great for predicting high risk loans. This model about the same as the Random Over Sampler.
       
   
-       Accuracy score

    ![Cluster accuracy](./Resources/cc_accuracy.png) 

-       Confusion matrix

    ![Cluster matrix](./Resources/cc_matrix.png) 

-       Imbalanced classification report

    ![Cluster report](./Resources/cc_report.png)

<br/>

## Results: Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

<br/>

-   SMOTEENN 


#### The accuracy even lower with the SMOTEENN model (54%). 79 of the 101 high risk category are correctly classified (Recall = 78%). This is an improvement over the previous 3 models. Again the precision (1%) is very low due to a very imbalanced data set with such high number of low risk clients. The F1 score is really low as well (0.02) which indicates a model that is not great for predicting high risk loans. The recall is getting a little better but the F1 score is not improving. 
       
    
-       Accuracy score
    ![SMOTEEN accuracy](./Resources/teen_accuracy.png) 

-       Confusion matrix
    ![SMOTEEN matrix](./Resources/teen_matrix.png) 

-       Imbalanced classification report
    ![SMOTEEN report](./Resources/teen_report.png) 


<br/>

## Results: Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

<br/>

-   Balanced Random Forest Classifier 

#### The accuracy has improved a bit with the Balanced Random Forest model (79%). 71 of the 101 high risk category are correctly classified (Recall = 70%). This is not too bad at finding the true high risk outcomes. Again the precision (3%) is very low due to a very imbalanced data set with such high number of low risk clients. The F1 score is really low  but getting better (0.06) which indicates a model that is not great for predicting high risk loans. The recall is getting a little better but the F1 score has only improved slightly. 
       
    
-       Accuracy score
    ![BRF accuracy](./Resources/rf_accuracy.png) 

-       Confusion matrix
    ![BRF matrix](./Resources/rf_matrix.png) 

-       Imbalanced classification report
    ![BRF report](./Resources/rf_report.png) 

-       Importance list report
    ![BRF report](./Resources/rf_import.png) 

<br/>

-   Easy Ensemble AdaBoost Classifier 
    
#### The accuracy has improved quite a bit with the Easy Ensemble model (93%). 93 of the 101 high risk category are correctly classified (Recall = 92%). This much better at finding the true high risk outcomes. Again the precision (9%) is very low due to a very imbalanced data set with such high number of low risk clients. The F1 score is low but has improved (0.16) which indicates a model that is a little better than all the other models for predicting high risk loans. The recall is quite good and the F1 score has improved slightly. 
       
-       Accuracy score
    ![EE accuracy](./Resources/ee_accuracy.png) 

-       Confusion matrix
    ![EE matrix](./Resources/ee_matrix.png) 

-       Imbalanced classification report
    ![EE report](./Resources/ee_report.png) 

<br/>

## Summary

<br/>

#### The Credit Risk data set requires addressing the class inbalance of the high risk and low risk loans. There are 101 high risk vs 17,104 low risk loans. It would be fairly easy to predict the low risk loans but it is of much more importance to a Credit company to determine high risk loans. 


#### The first two models (RandomOverSampler and SMOTE) used over sampling techniques. The results are models with low F1 scores and only moderate recall. The SMOTE model may also be sensitive to outliers. The third model (ClusterCentroid) utilized undersampling of the low risk category. This model did not do much better than the oversampled models.

#### The fourth model used a combinatorial approach of over- and undersampling (SMOTEENN). The accuracy was not improved but the recall did improve of the previous three models. The F1 score was very low though. 



#### The final two models used Ensemble Classifiers: Balanced Random Forest Classifier and Easy Ensemble Classifier. Both of these employ bootstrap sampling to achieve the balances samples. The Random Forest classifier showed an improvement in recall but not much improvement of the F1 score. The Easy Ensemble classifier showed the best results of all six models with very high accuracy and recall. The F1 score is a weighted average of precision and recall and the relative contribution of each is equal. The F1 score is not great with all the models but in this case the trade off of recall is more important than precision. It is more important to detect the high risk loans. I'd recommend the Easy Ensemble as a good start to identify high risk loans. But I'd also recommend redoing the models with scaling all the features. There is a large range of measurements within the features. Income and loan amount are very large numbers while features like interest rate are very small numbers. This may improve the recall even more with the Easy Ensemble classifier.