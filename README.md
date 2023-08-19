# 🗽Titanic Tragedy Analysis

![k-mitch-hodge-y-9-X5-4-vU-unsplash](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/bdc6c95c-9ee6-41da-a429-3cb51a7e7a3b)

# Python Libraries Used

- pandas
- numpy
- matplotlib
- statsmodel
- scikit-learn

## Objective

Our goal is to find the predictors for survivability during the Titanic tragedy using machine learning.


## About the dataset

The dataset records attributes from 888 passengers. It features:

- Passenger ID
- Survived
- Passenger class
- Name
- Sex
- Age
- Siblings aboard
- Parents aboard

The dataset can be found on the Stanford University [website](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html).

# I/Defining the goal:

The Titanic was a British passenger liner that sank in 1912. Carrying more than 2 200 passengers on board, the boat never made it to its final destination, New York City.
This event had a massive impact on our modern culture. The infamous James Cameron movie always made me wonder about the nature of the profiles who survived.<br>
Who were they?

Today, we will answer this question by analyzing the given data.

What made the Titanic survivor so special? 

Using diverse machine learning techniques, this project will aim at defining the best and worst predictors of survival according to the dataset. 

# II/Data Wrangling: 

After importing libraries and setting the directory.<br>
Our first task is to look for missing values. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/5de51b9d-c579-4822-af63-5180d0ba7f07)



Since the data has a binary outcome, we will use logistic regression.
In order to prepare our data for the regression we need to convert some string value into numerical. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/24fa8bac-4510-4fa3-a260-0aa4abbedbc2)





# III/ Analysis:
### A/Overall Analysis
In order to survive in a stressful situation some factors are more important than others.<br>
Our goal is to determine which feature will be detrimental to the survival of these passengers. 

Let's first take a look at the first feature, age: 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/f5b72b11-ee6f-4d67-9d33-21363340921d)

As we can see, most of the individuals were between 15 and 35 years old. Quite a young crowd.

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/989552b9-05d6-43ac-bdb5-6cc8775e055b)

Now, let's take a look at gender disparities. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/3b801cc6-0548-4859-b623-9a4a8749f416)

It seems that most passengers were males.<br>
Since the Titanic was a cruise ship per se, individuals had the choice between 3 classes. The first class is the most luxurious one.<br>
Let's move on to our next attribute, class: 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/7f934102-631a-489a-969f-f4fcb8ab8106)

We can observe without surprise that most passengers were in the 3rd class which is the most affordable. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/5cb0d6b1-a1c1-4383-8a50-172a821a6b31)

Regarding the amount shown on the graph, it is important to note that 80USD in 1912 is more or less equal to 2500USD in 2023. Quite an expense.<br> 
After taking a look at some important features. A question arises, do those attributes have that much of an impact on personal survival? 

Gender seems to have a detrimental impact on survival. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/62b2b9b0-4c81-4680-a19b-3e1253232434)

74% of the females survived. While only 19% of the males survived. As seen previously most of the passengers were males and yet they remained a minority. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/119d9691-628a-4fe9-9350-5f045ac1c18c)

Another factor of survival is the class the passenger was in. We can see that being in the first class is a significant advantage with 63% of individuals surviving. 
Since the odds of surviving in first class are quite important, we must check the gender distribution in this class in order to avoid any data bias. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/32a29f85-dae4-474e-8814-aabe8830e234)

The gender distribution in the first class is as excepted, males are not a minority. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/e290b29b-1125-4c87-b931-0e123189c501)

From the data analyzed in this part, we can already draw conclusions on the best and worst profiles.<br>
It seems that the best profile for survival would be to be a young wealthy female.<br>

Let's see if building a machine-learning model will change our expectations.

### B/Models

To optimize our model building, we will create two models with different attributes to maximize our accuracy.<br>
Once both models are evaluated, we will choose the most accurate one and use it to find the best and worst profiles. 

Using the logistic regression from Statsmodel

| Model| Independant Variable| Test Accuracy| 
|---|---|---|
| 1 | Pclass + Sex + Age + Fare + S_Ob + Pc_Ob + Embarked_P | 79%  |  
| 2 | Pclass + Sex + Age + S_Ob + Pc_Ob | 78%  |

Confusion matrix for the test sample model 1
![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/cdd40409-ab2c-4986-9607-335f9cbda872)

Confusion matrix for the test sample model 2

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/cc79497e-f5ba-4efa-93da-339583467c1f)


The first model is more accurate and has a better confusion matrix. We will therefore choose the first model. 
To find the best profiles we will use the coefficient from the Regression Results table. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/37feaac4-994b-4b6f-831c-d5556773c89c)

Using the formula for linear regression: Y=mX + b and the solver feature in Excel. 

# Conclusion







