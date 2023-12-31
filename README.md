# 🗽Titanic Tragedy Analysis

![k-mitch-hodge-y-9-X5-4-vU-unsplash](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/bdc6c95c-9ee6-41da-a429-3cb51a7e7a3b)

# Summary:
- Python Libraries Used
- Objective
- About the dataset
- I/Defining the goal
- II/Data Wrangling
- III/ Analysis
- Conclusion
- [Dashboard](https://github.com/Bruc3U/Titanic_analysis/blob/main/dashboard_titanic_pdf.pdf)


# Python Libraries Used

- pandas
- numpy
- matplotlib
- statsmodel
- scikit-learn

## Objective

Our goal is to use machine learning to identify predictors for survivability during the Titanic tragedy. 


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
This event had a massive impact on our modern culture. The infamous James Cameron movie always made me wonder about the nature of the individuals who survived.<br>
Who were they?

Today, we will answer this question by analyzing the given data.

What made the Titanic survivor so special? 

Using diverse machine learning techniques, this project will aim at defining the best and worst predictors of survival according to the dataset. 

# II/Data Wrangling: 

After importing libraries and setting the directory.<br>
Our first task is to look for missing values. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/5de51b9d-c579-4822-af63-5180d0ba7f07)



Since the data has a binary outcome, we will use logistic regression.

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/24fa8bac-4510-4fa3-a260-0aa4abbedbc2)

In order to prepare our data for the regression we need to convert string values into numerical ones.<br>
The table below will summarize every change made to the data: 

| String Value Conversion Table  | 
|---|
| male = 1 |
| female = 0 |
| Embarked_P C = 1 |
| Embarked_P S = 2 |
| Embarked_P Q = 3 |

Embarked signifies the port of embarkment for each passenger, we had to modify the value for all of the 3 ports.<br>
The data is finally ready for the next step, analysis. 

# III/ Analysis:
### A/Overall Analysis
Our goal is to determine which feature will be detrimental to the survival of these passengers. 

Let's take a look at the first feature, age: 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/f5b72b11-ee6f-4d67-9d33-21363340921d)

As we can see, most of the individuals were between 15 and 35 years old. Quite a young crowd.

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/989552b9-05d6-43ac-bdb5-6cc8775e055b)

Now, let's take a look at gender disparities. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/3b801cc6-0548-4859-b623-9a4a8749f416)

It seems that most passengers were males.<br>

Let's move on to our next attribute, class: 

The Titanic offered 3 classes, each passenger would get a different treatment depending on their class.<br>
For instance, the first class resembled more of a high-end luxury trip with private bedrooms and bathrooms.<br>
Whereas, the third-class passengers had to share bathrooms and slept in dormitory-style cabins. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/7f934102-631a-489a-969f-f4fcb8ab8106)

We can observe most passengers were in the 3rd class which is the most affordable. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/5cb0d6b1-a1c1-4383-8a50-172a821a6b31)

Regarding the amount shown on the graph, it is important to note that 80$ in 1912 is more or less equal to 2 500$ in 2023. Quite an expense.<br> 
After taking a look at some important features. A question arises, Do these attributes have much of an impact on survival? 

Gender seems to have a detrimental impact on survival. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/62b2b9b0-4c81-4680-a19b-3e1253232434)

74% of the females survived. While only 19% of the males survived. As seen previously most of the passengers were males and yet they remained a minority here. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/119d9691-628a-4fe9-9350-5f045ac1c18c)

The passenger's class also plays a role in determining survival rates. We can see that being in the first class, is a significant advantage with 63% of individuals surviving. 
Since the odds of surviving in first class are quite important, we must check the gender distribution in this class in order to avoid any data bias. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/32a29f85-dae4-474e-8814-aabe8830e234)

The gender distribution in the first class follows the main trend. Males are the majority. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/e290b29b-1125-4c87-b931-0e123189c501)

From the data analyzed so far, we can already draw conclusions on the best and worst profiles.<br>
It seems attributes such as gender, age, and class have an impact on survival.<br>
For instance, a wealthy female in the first class will have a better chance of surviving compared to an older male in the third class.

In order to elaborate on this hypothesis, we will build a logistic regression model.

### B/Models

We will create two models with different features to maximize our accuracy.<br>
Once both models are evaluated, we will choose the most accurate one and use it to find the best and worst profiles. 

Using the logistic regression from Statsmodel

| Model| Independant Variable| Test Accuracy| 
|---|---|---|
| 1 | Pclass + Sex + Age + Fare + S_Ob + Pc_Ob + Embarked_P | 79%  |  
| 2 | Pclass + Sex + Age + S_Ob + Pc_Ob | 78%  |

Both models will predict the survival column, which in binary, tells us which individual survived. 

Confusion matrix for the test sample model 1<br>

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/cdd40409-ab2c-4986-9607-335f9cbda872)

Confusion matrix for the test sample model 2<br>

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/cc79497e-f5ba-4efa-93da-339583467c1f)

The first model is more accurate and has a better confusion matrix. We will therefore choose the first model. 
To find the best profiles we will use the coefficient from the Regression Results table. 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/37feaac4-994b-4b6f-831c-d5556773c89c)

The formula for linear regression is a useful tool to predict values using the data from the Regression Results table.<br>
Linear regression: Y=mX + b 

This formula is needed to understand the relationship between coefficients (weights). This will help us compute a Final Survival Score for each profile.<br>
The higher the score the greater chance the individual will have to survive. 
Let's look at some examples: 

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/880dbe1e-c951-4844-ac08-36622390c9dd)

The poor old man profile and the poor old woman profiles, explain what we explored earlier in our first observations.<br>
Both profiles are identical, except for the gender.<br>
As you can see, the poor old woman has a drastically better survival score.<br>
It proves that gender is one of the most important survival predictors.

Another important predictor would be the class. Both Sex and Class have a coefficient of respectively -2.8 and -1.3.<br>
Other coefficients are closer to  0, which illustrates their diminished weight on the survival score. 

The two profiles named Worst Profile and Best Profile were conducted by the Excel solver. We can see the similarity with the previous estimation.

![image](https://github.com/Bruc3U/Titanic_analysis/assets/142362478/5f58a158-cb01-444c-979a-7c3564eafee3)

For the worst profile, the port of departure number 3 was chosen, and the best profile chose port number 1.  We can observe that a lower age and a higher class/ticket fare give more chances of survival as stated earlier.
On another note, the worst profile chooses to have 2 children and 2 parents, this proves the negative aspect of those features. 



# Conclusion
The Titanic dataset offers some interesting insights into what features are the most detrimental to survival.<br>
To survive the Titanic tragedy, one would have more chances if they were a female boarding in Cherbourg France, as young as possible, traveling solo with a high ticket fare and class.<br>
Alternatively, the most unfavorable situation would be to embark from Queenstown, England, while being a male in advanced age, accompanied by both children and parents and occupying the lowest class.









