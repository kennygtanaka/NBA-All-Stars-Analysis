# NBA-All-Stars-Analysis
Final Project (creative/open-ended) for MTH123 "Math for Data Science" course at Tufts University. NBA All-Star player data was analyzed through utilization of PCA, Sparse PCA and clustering. 

## Python Libraries Used 
* NumPy : For array and matrix operations
* Pandas : For working with dataframes 
* sklearn : For PCA and regression models
* matplotlib: For plotting data 

## Data Used 
* Kaggle Dataset: NBA All Star Game 2000-2016 - www.kaggle.com/fmejia21/nba-all-star-game-20002016
* Kaggle Dataset: NBA Players - www.kaggle.com/justinas/nba-players-data

## Abstract
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This paper will be looking into the classification of NBA all star players based on their season statistics. It will find the best feature transformation and classifier that has the highest accuracy for correctly classifying all star players from their statistics. The second problem the paper will be addressing is if all star players can be separated into groups based on their characteristics. This paper will be focusing on Sparse PCA and PCA feature representations for classification and visualization. 

## 1 Introduction

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The National Basketball Association (NBA) is the world’s most elite basketball league. It houses the top 400 players on the planet and brings in billions of dollars of revenue yearly. Every year out of the 400 players in the league, 24 players are chosen to be all stars. These all star players are regarded as the best players who contribute most to their teams and to winning basketball games. However every basketball player isn’t the same. They play a variety of positions and have different physical attributes such as height and weight. Throughout this project I aim to answer the question: What kind of players become all star players? 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The data being used for this project is two different sets from Kaggle. The first dataset is a dataset based on each player’s information and statistics. Examples for the variables included in this dataset are height, weight, points per game (PPT), assists per game (AST), rebounds per game (REB) and many others. The other dataset is information on players who made the all star team. This dataset will only be used to denote whether the player was an all star that season. [2][3]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I took a few steps to clean the data for further analysis. I first dropped variables that I thought wouldn’t be related to whether a player made the all-star team that season. Some examples of these variables are: season, draft number, college and country. From there I matched the names from the all-star dataset and made a column of binary values (1 for all-star 0 for not all-star). I will be using this column as the output data to compare my predictions against. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After cleaning the data, I explored the data to see if there are any variables that are correlated with making the all star team. To get started, I first made a scatterplot showing height vs weight. 

![alt text](/images/image_1.png?raw=true)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The yellow dots denote that the player was an all star. The plot shows that being taller and weighing more does not necessarily correspond to making the all star team. The yellow dots are scattered throughout the different heights and weights. The next variable I tried plotting was net rating vs. age. Net rating is how many points the player contributes on average when they are playing. 

![alt text](/images/image_2.png?raw=true)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This plot also shows that the two variables chosen do not necessarily correlate with whether a player was an all star or not. The last set of variables I tried plotting to explore the data would be most perceived to have correlation with whether they made the all star team, points and assists. 

![alt text](/images/image_3.png?raw=true)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;As predicted, there are many all star players that make up the right portion of the scatter plot that signifies high amounts of points and assists. However, there are a few outliers, the most questionable one being the yellow dot at around 12 pts and less than two assists on average. These numbers would regularly be one that a regular player in the league averages, how did this player make the all-star team? To answer these questions I did a further analysis of all the variables. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Throughout the project I utilized a dimensional reduction technique that we hadn’t gone over in class before. This is Sparse PCA. In order to understand Sparse PCA, it may be helpful to define a sparse matrix. A sparse matrix is one that is composed of mainly zeros. Similarly in Sparse PCA, less variables are represented and more zeros are present. PCA is a linear combination of all the variables whereas Sparse PCA is only a combination of a select few of these. [1]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Some advantages of Sparse PCA is that it may be easier to interpret which variables affect the data the most since all variables aren’t being considered. Mathematically, the Sparse PCA optimization problem is similar to the PCA optimization problem except it has one extra constraint. This extra constraint controls the number of non-zero loadings that need to be less than a value k. The higher the k value is, the less variables are part of the linear combination. If the k value is equal to the number of variables, this would be the same as PCA. I will be using Sparse PCA to address both of the problems, utilizing it as a feature transformation and visualization tool. [1]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project yielded successful results as I was able to successfully classify what kind of players become allstars through classification and clustering. I will first highlight the problems that need to be solved, go over the method in which they were solved, and lastly go over the results and proper conclusions. 

## 2 Problem 


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Throughout this project there are two problems that I try to solve. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The first problem is, which feature transformation and classifier can I use on player statistics to predict whether a player becomes an all star that season? To answer this problem we will be looking for a feature transformation and classifier combination that is able to predict whether a player becomes an all star at a high accuracy rate. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The second problem is, what kind of players become all stars? Can we categorize different types of all star players? This problem will be successfully answered if I am able to find clear clusters within the all star players data. 



## Process and Results 

Displayed in Jupyter Notebook 
