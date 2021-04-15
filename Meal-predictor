
#Meal Plan suggestion Functions
import pandas as pd

def trend_filter(trend,predictionvalue):
    indexcode = "error"

    if trend == "stable":
        indexcode = 0; #stable-user can eat a low gi food if they want
    if trend == "rapidly-increasing" and predictionvalue[1] in range(180,250) and predictionvalue[0] in range(90,180):
        indexcode = 1; #high --> workout ie active engangement
    if trend =="rapidly-increasing" and predictionvalue[1] > 300 and predictionvalue[0] in range(100,180):
        indexcode = 2 ; #dangerously high --> workout ie active engangement --> consult doctor
    if trend =="rapidly-increasing" and predictionvalue[1] < 200 and predictionvalue[0] in range(100,180):
        indexcode = 3; #stable-high but no to extreme level --> don't eat any food
    if trend =="stable-increasing" and predictionvalue[1] < 200 and predictionvalue[0] in range(100,180):
        indexcode = 3; #stable-high but no to extreme level --> don't eat any food
    if trend =="stable-increasing" and predictionvalue[1] > 300 and predictionvalue[0] in range(200,400):
        indexcode = 2; #stable-high but no to extreme level --> don't eat any food
    if trend =="stable-increasing" and predictionvalue[1] > 200 and predictionvalue[0] > 200:
        indexcode = 1; #Sugar Level is ghigh
    if trend == "rapidly-decreasing" and predictionvalue[1] < 90 and predictionvalue[0] in range(150,250):
        indexcode = 4; #eat upper high gi food
    if trend == "rapidly-decreasing" and predictionvalue[1] < 120 and predictionvalue[0] in range(150,250):
        indexcode = 5; #eat mid high gi food
    if trend == "rapidly-decreasing" and predictionvalue[1] > 120 and predictionvalue[1] < 200 and predictionvalue[0] > 150 and predictionvalue[0] < 200:
        indexcode = 3; #don't eat anything
    if trend == "stable-decreasing"and predictionvalue[1] > 120 and predictionvalue[1] < 200:
        indexcode = 6 # eat low gi food
    if trend == "stable-decreasing"and predictionvalue[1] < 120:
        indexcode = 5 # eat mid high gi food
    if trend == "stable-decreasing" and predictionvalue [1] > 200:
        indexcode =1; #active engagement

    return indexcode



def filter_function(foodtype,glycemicdata):
    if foodtype == "mid":
        glnew = glycemicdata.query('Index >= 50'and 'Index <= 69')
    elif foodtype == "low":
        glnew = glycemicdata.query('Index <= 49' and 'Index >= 20')
    elif foodtype == 'upper-high':
        glnew = glycemicdata.query('Index >= 80'and 'Index <= 100')
    elif foodtype == 'lower-high':
        glnew = glycemicdata.query('Index < 80' and 'Index >= 70')

    glnew = glnew.drop(['Index'], axis = 1)
    return glnew


def message_function(indexcode):
    raw_dataset = pd.read_csv('SF-GlycemicData(CSV).csv')
    message = ""
    notification = ""

    glycemicdata = raw_dataset.copy()
    glycemicdata.isna().sum()
    glycemicdata = glycemicdata.dropna()
    if indexcode == 5:
        message = 'You sugar level is reaching low levels consume one of the following items within the next 30 minute interval'
        notification = filter_function('lower-high',glycemicdata)
    elif indexcode == 6:
        message = 'You sugar level is fluctating a bit consume one of the following items to stabelize it within the next 30 minute interval'
        notification = filter_function('low',glycemicdata)
    elif indexcode == 4:
        message = 'You sugar level is reaching very low levels consume one of the following items to raise it within the next 30 minute interval'
        notification = filter_function('upper-high',glycemicdata)
    elif indexcode == 0:
        message = 'You sugar level is stable you may consume one of the following items if you so please over the next 30 minute interval'
        notification = filter_function('mid',glycemicdata)
    elif indexcode == 3:
        notification = "Your sugar is stable but on the brink of sustansive change Do not eat anything until further notice: Check Back in 30 mins"
    elif indexcode == 1:
        notification = " Your Sugar is projected to stay at a very high level Workout or take walk, burn some calories."
    elif indexcode == 2:
        notification = "Your Sugar is projected to reach dangerously high level please seek medical help"


    fullst = [message,notification]
    return fullst

