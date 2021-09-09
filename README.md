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

## Process and Results 

Displayed in Jupyter Notebook 
