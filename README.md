# KNN with hand written Hebrew letters
A script for running KNN on handwritten Hebrew letters reaching an accuracy of classifying each letter with 76% accuracy


In order to run the script you need to run it in the terminal
the scripts needs a path for the dataset of the hebrew letters and if the path is not given then the script will automatically 
search in the working diractory .
The script uses the Chi Sqaure distance function and K=7 because thats what gave the best reasults at the time of testing.
you are free to use the euclidean distance function and any other K by using the arguments --metric and --k  at the same time.
use --help for more information.

Dataset reference
[1] I. Rabaev, B. Kurar Barakat, A. Churkin and J. El-Sana. The HHD Dataset. The 17th International Conference on Frontiers in Handwriting Recognition, pp. 228-233, 2020.
