# K-means
- Simple K-means algorithm
- Assignment 2 of COMP527
- Corresponding data can be download at [here](http://cgi.csc.liv.ac.uk/~danushka/lect/dm/CA2data.txt)

# Manual
1. Replace the data file path
Before executing the "K_means.py", the path of the data files should be modified - towards the **line 11** as shown below:
```
file="Absolute path of CA2data.txt here" 
```
In order to run the script, please replace the content between "" to the absolute path of the data file. After the replacement, the path of file may like this:
```
file="/Users/Desktop/DA2/CA2data.txt"
```

2. Run the script
In Unix(Mac OS/Linux) Terminal with the Python 3 environment and the library of numpy, matplotlib, random and opertaor. After switching into the directory of the source code file, input:   
```
python K_means.py
```
or  
```
python3 K_means.py
```   
to run the script. 

# Warning 
If you are the student of UoL, *DON'T COPY ANY LINE OF THIS CODE*. Or you'll recieve a terrible mark. 
Don't ask me why I know this. :P

The discussion about using data point instead of mean as the centroid is very interesting. You can read my idea at: [here](https://typewind.github.io/2017/03/27/instance-output/)
