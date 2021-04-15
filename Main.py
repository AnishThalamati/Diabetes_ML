
#Data on Glucose
from mealsuggestion import *
from mlscript import *

from datetime import datetime
import pytz


tz_NY = pytz.timezone('America/New_York')
datetime_NY = datetime.now(tz_NY)
StandardTime = datetime_NY.strftime("%H:%M:%S")
print(StandardTime)


def Convert(string):
    li = list(string.split(":"))
    return li


lsttime = Convert(StandardTime)

newhour = int(lsttime[0])
newminute = int(lsttime[1])

custom = input("Would you like to enter a custom time(Yes) or use the current time(No): ")

if (custom == "Yes"):
    hour = int(input("Enter the Hour: "))
    minute = int(input("Enter the Minute: "))

if (custom == "No"):
    hour = newhour
    minute = newminute


newdf = trend_function(hour,minute)
trend = newdf[0]
predictionvalue = newdf[1]

print("The trend of your sugar levels over the next 30 minute interval is: ", trend)
print("Your sugar level right now is: ", predictionvalue[0], "and is the 30 minutes your sugar level is projected to be: ",predictionvalue[1])


indexcode = trend_filter(trend,predictionvalue)
print("The corresponding method code to your trend is:", indexcode)

custommessage = message_function(indexcode)
print(custommessage[0])
print(custommessage[1])
