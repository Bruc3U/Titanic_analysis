# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:51:55 2023

@author: 98yan
"""

%cd Y:\Document\Yanis\Professionel\Portfolio\titanic

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

df = pd.read_csv('titanic.csv',delimiter=';')
df.info()


#count of male/female

av9 = df.loc[:,['Sex','Passenger_Id']]
av10 = pd.DataFrame(av9.groupby('Sex').agg({'Passenger_Id':'count'}))  

couleurs = ['palevioletred','cornflowerblue']
plt.bar(av10.index,av10['Passenger_Id'],color=couleurs)
plt.title('Count of Male/Female')
for index, value in enumerate(av10['Passenger_Id']):
    rounded_value = round(value, 2)
    plt.text(index,value,str(rounded_value))

#Age of the passengers

av7 = df.loc[:,['Age','Passenger_Id']]
av8 = pd.DataFrame(av7.groupby('Age').agg({'Passenger_Id':'count'}))


colormap_name = 'viridis' 
colormap = matplotlib.colormaps[colormap_name]
num_colors = len(av8.index)
colors = colormap(np.linspace(0, 1, num_colors))

plt.bar(av8.index,av8['Passenger_Id'], color=colors)
plt.xticks(range(0, len(av8.index), 5)) 
plt.title('Age of the Titanic passenger')
plt.xlabel('Age')
plt.ylabel('Count')


#let's segment the Age data

age_bins = [0,15,35,80]
age_labels= ['0-15','15-35','35-80']

df2 = df.copy()
df2['Age_Segment'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

segment_counts = pd.DataFrame(df2['Age_Segment'].value_counts().sort_index())

colormap_name = 'Accent' 
colormap = matplotlib.colormaps[colormap_name]
num_colors = len(segment_counts.index)
colors = colormap(np.linspace(0, 1, num_colors))
plt.bar(segment_counts.index,segment_counts['Age_Segment'],color=colors)
plt.title('Age Segments')
for index, value in enumerate(segment_counts['Age_Segment']):
    rounded_value = round(value, 2)
    plt.text(index,value,str(rounded_value))
    
    
#survival rate per sex 

av5 = df.loc[:,['Survived','Sex']]
av6 = pd.DataFrame(av5.groupby('Sex').agg({'Survived':'mean'}))

couleurs = ['palevioletred','cornflowerblue']
plt.bar(x=av6.index,height=av6['Survived'],color=couleurs)
plt.title('Average Survival rate per Sex')
plt.xlabel('Classes')
plt.ylabel('Survival Rate')
plt.text(0,0.74,'74%')
plt.text(1,0.19,'19%')


#count of individuals per class
av11 = df.loc[:,['Pclass','Passenger_Id']]   
av12 = pd.DataFrame(av11.groupby('Pclass').agg({'Passenger_Id':'count'}))

couleurs= ['greenyellow','olive','grey']
plt.bar(av12.index,av12['Passenger_Id'],color=couleurs)
plt.xticks([1,2,3])
plt.title('Number of individual per class')
plt.xlabel('classes')
for index, value in enumerate(av12['Passenger_Id'],start=1):   
    plt.text(index,value,str(value))

#average fare per class

av = df.loc[:,['Pclass','Fare']]
av1 = av.groupby('Pclass').agg({'Fare':'mean'})
av2 = pd.DataFrame(av1)

couleurs = ['darkcyan','olive','grey']
plt.bar(x=av1.index,height=av1['Fare'],color=couleurs)
plt.title('Average Fare per Class')
plt.xticks([1,2,3])
plt.xlabel('Classes')
plt.ylabel('Fare in USD')
for index, value in enumerate(av1['Fare'],start=1):
    rounded_value = round(value, 2)
    plt.text(index,value,str(rounded_value) + '$')

#Survival rate among classes

av3 = df.loc[:,['Pclass','Survived']]
av4= pd.DataFrame(av3.groupby('Pclass').agg({'Survived':'mean'}))

couleurs = ['yellowgreen','wheat','peru']
plt.bar(x=av4.index,height=av4['Survived'],color=couleurs)
plt.title('Average Survival rate per Class')
plt.xticks([1,2,3])
plt.xlabel('Classes')
plt.ylabel('Survival Rate')
for index, value in enumerate(av4['Survived'],start=1):
    rounded_value = round(value, 2) * 100 
    plt.text(index,value,str(rounded_value) + '%')
    


#survival rate per age 

survival_rate = df2.groupby('Age_Segment').agg({'Survived':'mean'})

colormap_name = 'Accent' 
colormap = matplotlib.colormaps[colormap_name]
num_colors = len(segment_counts.index)
colors = colormap(np.linspace(0, 1, num_colors))

plt.bar(survival_rate.index,survival_rate['Survived'],color=colors)
plt.title('Survival rate per Age group')
plt.ylabel('Survival Rate in %')
plt.xlabel('Age segments')
for index, value in enumerate(survival_rate['Survived']):
    rounded_value = round(value, 2) * 100
    plt.text(index, value,str(rounded_value) + '%')
    
#Gender on the first class

av13 = df.loc[:,['Sex','Pclass'] 
av14 = pd.DataFrame(av13[av13['Pclass'] == 1])
av15 = pd.DataFrame(av14.groupby('Sex').agg({'Pclass':'count'}))

couleurs = ['palevioletred','cornflowerblue']
plt.bar(av15.index,av15['Pclass'],color=couleurs)
plt.title('Gender Repartition in First Class')
for index, value in enumerate(av15['Pclass']):   
    plt.text(index, value,str(value))
    