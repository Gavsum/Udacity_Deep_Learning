# TV Script Generation
## Overview
In this project we attempted to generate realistic TV scripts for the Simpsons. We fed 27 seasons worth of Simpsons scripts as training data for an RNN, and then used it to generate a script for a scene at Moe's Tavern. 
![](https://i.imgur.com/Expmhum.jpg)

## Model
To preprocess the data we tokenized punctuation, and created a word-embedding lookup table. We then used a Recurrent Neural Network architecture which consists of stacked LSTM cells. The LSTM cells add temporality to the model's predictions, such that it considers all of the text which precided its current prediction. 
![](https://i.imgur.com/ntO4Fb4.png)

## Results
Hilarity ensues.
```
homer_simpson: so not too guy? well, you little lease i ever 
got us bar no one american your brilliant man!

homer_simpson:(sobs) oh thank you, you mr. this unsourced, 
homer. The greatest year of the thing you got their loser?
Get up my pin barkeep, the sturdy sittin' in the game!

homer_simpson: i've had interested out of the game at my new
game is a car, moe.

crowd: yeah, you knockin' beer?

barney_gumble: how happened to you? Man, you can't do so 
people more.

moe_szyslak: okay, i'm like some american comedy somethin'?

fox_mulder: win this people were getting years.
(to vicious) but i like to find salad house out of the bar.

lenny_leonard: no way and just do i see the great year 
into a way to be just leaving, ladies up for my bar.

barney_gumble: oh, i don't want anything to be fun.

homer_simpson: i'm gonna get them to turn around out for 
all my favorite delighted down his face is
```
