## Project Title
### Disaster Response Pipeline Project

## Project Description
1. This project works on disaster data from Figure Eight.
2. It builds a model for an API with ability to classify disaster messages.
3. User interface for this project will be a falsk web app.


### Installation:
Packages included in Anaconda distribution.

### Motivation:
1. Motivation for this project is to give the user ability to quickly identify the category of the disaster message.
2. With this ability the response times will be greatly improved.

### Files:
Data files are as follows:

1. disaster_categories.csv    
2. disaster_messages.csv

Python code:
1. process.py
2. train_classifier.py
3. run.py

### Author
Saurabh Daphtardar


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
