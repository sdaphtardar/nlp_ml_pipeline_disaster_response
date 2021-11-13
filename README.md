## Project Title
#### Disaster Response Pipeline

### Project Description
1. This project works on disaster data from Figure Eight.
2. It builds a model for an API with ability to classify disaster messages.
3. User interface for this project is a falsk web app.


### Installation:
Packages included in Anaconda distribution.

### Motivation:
1. Objective for this project is to give the user ability to quickly identify
the category of disaster message.
2. With this ability the response times will be greatly improved.

### Files:
Data files are as follows:

1. disaster_categories.csv = file containing categories of each message    
2. disaster_messages.csv = file containing message and genre

Flask template files:
1. go.html = This file is used to display main page of the web app
2. master.html = This file is used to display classification result of the input query

Database file:
1. DisasterResponse.db = This is an intermediary file created by process_data.py
containing the database table which is read by train_classifier.py

Machine learning model file:
1. classifier.pkl = This is a model file created by train_classifier.py

Python code:
1. process_data.py = This file performs needed wrangling of the input data and
saves the results in a database table.
2. train_classifier.py = This file reads data from the database table, trains a
model on it and saves it.
3. run.py = file used to run the app

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
