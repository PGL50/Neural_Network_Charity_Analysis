# Overview of Neural Network Charity Analysis


### Machine learning and neural networks will be used for the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The neural network results will be saved in an HDF5 file. After the first run the model will be attempted to be optimized 3 different ways to increase the accurace of the model.

<br/>

## Results: Data Preprocessing

<br/>

-   What variable(s) are considered the target(s) for your model? 

    #### The binary variable "IS_SUCCESSFUL" is the target feature for the model. So the model wil be predicting is the money was used effectively.

``` python
    y = application_df["IS_SUCCESSFUL"].values
```

-   What variable(s) are considered to be the features for your model? 

    #### The feature variables under consideration for the model are APPLICATION_TYPE, AFFILLIATION, ClASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.

    #### The two columns with mode than 10 categories (APPLICATION_TYPE and CLASSIFICATION) were binned into fewer categories.

<br/>

-   APPLICATION_TYPE categories

    ![APP classes](./Resources/appFirst.png) 

<br/>

-   CLASSIFICATION categories
![CLASS classes](./Resources/classFirst.png) 

<br/>

-   What variable(s) are neither targets nor features, and should be removed from the input data? 

    #### The variables EIN and NAME are not considered as features or targets. They are names and ID's that don't add any information as to whether the money use is successful.

<br/>

## Results: Compiling, Training, and Evaluating the Model

<br/>

-   How many neurons, layers, and activation functions did you select for your neural network model, and why?

    #### The number of neurons for the initial model is 45 for the first hidden layer and 20 for the second hidden layer. This was a starting point based on the number of input features after using OneHotEncoder to transform the categorical variables. The most common activation function was 'relu' so that was used as the starting function. The output function used was 'sigmoid' which is a common starting point in the Module.

```python
    # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
    number_input_features = len(X_train[0])
    hidden_nodes_layer1 = 45
    hidden_nodes_layer2 = 20

    # First hidden layer
    nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, 
                          activation="relu"))

    # Second hidden layer
    nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

    # Output layer
    nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```

-   Were you able to achieve the target model performance (75% accuracy)?

    #### The target perfomance was not achieved.

<br/>

-   Deliverable 1 model achieved 72.9% accuracy.

    ![Model 1](./Resources/model1.png) 

<br/>

-   Deliverable 2 model achieved 70.5% accuracy.

    ![Model 2](./Resources/model2.png) 

<br/>

-   Deliverable 3 model achieved 70.7% accuracy.

    ![Model 3](./Resources/model3.png) 

<br/>

-   Deliverable 4 model achieved 70.7% accuracy.

    ![Model 4](./Resources/model4.png) 


-   What variable(s) are neither targets nor features, and should be removed from the input data? 

    #### The variables EIN and NAME are not considered as features or targets. They are names and ID's that don't add any information as to whether the money use is successful.

<br/>



<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

#### The Credit Risk data set requires addressing the class inbalance of the high risk and low risk loans. There are 101 high risk vs 17,104 low risk loans. It would be fairly easy to predict the low risk loans but it is of much more importance to a Credit company to determine high risk loans. 


#### The first two models (RandomOverSampler and SMOTE) used over sampling techniques. The results are models with low F1 scores and only moderate recall. The SMOTE model may also be sensitive to outliers. The third model (ClusterCentroid) utilized undersampling of the low risk category. This model did not do much better than the oversampled models.

#### The fourth model used a combinatorial approach of over- and undersampling (SMOTEENN). The accuracy was not improved but the recall did improve of the previous three models. The F1 score was very low though. 



#### The final two models used Ensemble Classifiers: Balanced Random Forest Classifier and Easy Ensemble Classifier. Both of these employ bootstrap sampling to achieve the balances samples. The Random Forest classifier showed an improvement in recall but not much improvement of the F1 score. The Easy Ensemble classifier showed the best results of all six models with very high accuracy and recall. The F1 score is a weighted average of precision and recall and the relative contribution of each is equal. The F1 score is not great with all the models but in this case the trade off of recall is more important than precision. It is more important to detect the high risk loans. I'd recommend the Easy Ensemble as a good start to identify high risk loans. But I'd also recommend redoing the models with scaling all the features. There is a large range of measurements within the features. Income and loan amount are very large numbers while features like interest rate are very small numbers. This may improve the recall even more with the Easy Ensemble classifier.