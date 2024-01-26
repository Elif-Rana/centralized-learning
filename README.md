# Centralized Learning
In machine learning, centralized learning refers to a training paradigm where all the data is stored and processed on a central server. This server then trains a single model that is used for all purposes.

The steps for centralized learning are as followed.
  - Data is collected from various sources and stored on the central server.
  - The central server trains a machine learning model on the collected data.
  - As result, a single integrated model is created.

In summary, the central server combines all the data from all users, extracts the features, and then trains a machine learning model using the combined data.

For more information about the process and comparison of three different machine learning strategies (individual, centralized, federated), you can check 
https://www.mdpi.com/2075-4426/12/10/1584

## Get Started
```
#clone the repository
git clone https://github.com/Elif-Rana/centralized-learning.git

#create virtual environment
python -m venv venv

#activate virtual env. (Linux based operating systems)
source venv/bin/activate

#activate virtual env. (Windows)
.venv\Scripts\activate

#install requirements
pip install -r requirements.txt

#run
python centralized.py --epochs 3
