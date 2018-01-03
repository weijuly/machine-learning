# kNN classification algorithm

The full name of the algorithm is k-Nearest-Neighbors. It works on the basic principle of neighborhood.

## working
The algorithm measures the distance ( spatial Euclidean distance ) between the unknown data points to all other data points. Then the distances are sorted based on the descending order, which reveals the closest neighbors of the unknown data point. The category to which the majority of the neighbors belong to is the category of the unknown data point. Quoting an example from the book on movie categorization:
```
MOVIE                           # OF KICKS  # OF KISSES GENRE
---------------------------------------------------------------
California Man                  3           104         Romance
He’s Not Really into Dudes      2           100         Romance
Beautiful Woman                 1           81          Romance
Kevin Longblade                 101         10          Action
Robo Slayer 3000                99          5           Action
Amped II                        98          2           Action
**Unknown**                     18          90          ???

```
Based on the above data, we can infer that the Unknown movie is likely a Romance movie, as it has higher number of kisses. So, we group it to the Romance Genre. Mathematically, based on distance calculations:
```
MOVIE                           DISTANCE TO Unknown
---------------------------------------------------
California man                  20.5
He’s Not Really into Dudes      18.7
Beautiful Woman                 19.2
Kevin Longblade                 115.3
Robo Slayer 3000                117.4
Amped II                        118.9
```
The closest neighbors based on distance are romance movies, so this unknown movie should be a romance movie.


## requirements 
This algorithm requires training in the first place, as it is a supervised learning. The training data usually is a well known classified data. For the sake of implementation we'll assume we have a matrix form of data along with it's classification

```
training-data.txt ( format: F1  F2  F3  F4  CT ):
23  45  32  44  A
12  62  48  37  B
56  46  33  95  C
25  48  43  42  A
50  37  39  32  B
...

```
The training data is a two dimensional matrix. Each row represents a piece of known data point, identified by the category in the last column. Each point carries the measurement of its features as a numerical value. 

## implementation

See [kNN.py](kNN.py)

## notes
1. For each classification, we calculate the distances from the unknown point to entire dataset, which is expensive
2. Discrete values like YES/NO can be transformed to 1/0