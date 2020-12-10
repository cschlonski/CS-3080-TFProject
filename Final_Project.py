
#Pulls the data from the api with a start and end date
def api_search(code,start,end):
    url="http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=6230647e6745477995c00355202410&q="+code+"&format=json&date="+start+"&enddate="+end +"&tp=24"
    return url

import json
import urllib.request
import re

#allows user to enter what they want to search
code=input('Enter a zip code:')
start=input('Enter a start date in the format yyyy-mm-dd:')
end=input('Enter an end date in the format yyyy-mm-dd:')
graph_type=input('Do you want the data displayed daily, monthly, or yearly?')

minTemp={}
maxTemp={}
windSpeed={}
humidity={}

#splits the start and end date into groups
dateRegex=re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)')
mo=dateRegex.search(start)

dateRegex=re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)')
mo2=dateRegex.search(end)

start2=start
month=int(mo.group(2))+1
if month>9:
    end2 = mo.group(1) + '-' + str(month) + '-' + mo.group(3)
else:
    end2=mo.group(1)+'-'+'0'+str(month)+'-'+'01'
j=2

#pulls the data from the api seperately since the max amount of dates is 35. This allows for unlimited time
while(end2!=end):
    url = api_search(code, start2, end2)
    obj = urllib.request.urlopen(url)

    rawdata = json.load(obj)

    data = rawdata['data']

    # pulls the data and places it into dictionaries
    for item in data['weather']:
        i = 0
        date = item['date']
        minTemp.update({date: item['mintempF']})
        maxTemp.update({date: item['maxtempF']})
        for item2 in item['hourly']:
            test = item['hourly']
            test2 = test[i]
            windSpeed.update({date: test2['windspeedMiles']})
            humidity.update({date: test2['humidity']})
            i = i + 1

    dateRegex = re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)')
    mo3 = dateRegex.search(end2)
    day = int(mo3.group(3)) + 1
    if day > 9:
        start2 = mo3.group(1) + '-' + mo3.group(2) + '-' + str(day)
    else:
        start2 = mo3.group(1) + '-' + mo3.group(2) + '-' + '0' + str(day)

    month = int(mo.group(2)) + j

    if month > 9:
        end2 = mo3.group(1) + '-' + str(month) + '-' + '01'
    else:
        end2 = mo3.group(1) + '-' + '0' + str(month) + '-' + '01'
    j = j + 1

    dateRegex = re.compile(r'(\d\d\d\d)-(\d\d)-(\d\d)')
    mo4 = dateRegex.search(start2)
    if (mo4.group(2) == '12'):
        end2 = str(int(mo4.group(1)) + 1) + '-01-' + '01'
        j = 1

    if (mo2.group(1) == mo4.group(1) and mo2.group(2) == mo4.group(2)):
        end2 = end

url = api_search(code,start2,end2)

obj = urllib.request.urlopen(url)

rawdata = json.load(obj)

data = rawdata['data']

print("\n")

#pulls the data and places it into dictionaries
for item in data['weather']:
    i=0
    date=item['date']
    minTemp.update({date:item['mintempF']})
    maxTemp.update({date:item['maxtempF']})
    for item2 in item['hourly']:
        test=item['hourly']
        test2 = test[i]
        windSpeed.update({date:test2['windspeedMiles']})
        humidity.update({date:test2['humidity']})
        i=i+1

finalMinTemp=[]
finalMaxTemp=[]
finalwindSpeed=[]
finalHumidity=[]

#puts the data into an array so it can be graphed. Will also average the data based on month or year if needed
if(graph_type=='daily'):
    for key, value in minTemp.items():
        finalMinTemp.append(int(value))
    for key, value in maxTemp.items():
        finalMaxTemp.append(int(value))
    for key, value in windSpeed.items():
        finalwindSpeed.append(int(value))
    for key, value in humidity.items():
        finalHumidity.append(int(value))

elif(graph_type=='monthly'):
    total = 0
    count=0
    temp = ''
    for key, value in minTemp.items():
        date=key
        dateRegex=re.compile(r'\W\d\d\W')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalMinTemp.append(average)
            total=int(value)
            count=1
    average = total / count
    finalMinTemp.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in maxTemp.items():
        date=key
        dateRegex=re.compile(r'\W\d\d\W')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalMaxTemp.append(average)
            total=int(value)
            count=1
    average = total / count
    finalMaxTemp.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in windSpeed.items():
        date=key
        dateRegex=re.compile(r'\W\d\d\W')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalwindSpeed.append(average)
            total=int(value)
            count=1
    average = total / count
    finalwindSpeed.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in humidity.items():
        date=key
        dateRegex=re.compile(r'\W\d\d\W')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalHumidity.append(average)
            total=int(value)
            count=1
    average = total / count
    finalHumidity.append(average)

else:
    total = 0
    count=0
    temp = ''
    for key, value in minTemp.items():
        date=key
        dateRegex=re.compile(r'\d\d\d\d')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalMinTemp.append(average)
            total=int(value)
            count=1
    average = total / count
    finalMinTemp.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in maxTemp.items():
        date=key
        dateRegex=re.compile(r'\d\d\d\d')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalMaxTemp.append(average)
            total=int(value)
            count=1
    average = total / count
    finalMaxTemp.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in windSpeed.items():
        date=key
        dateRegex=re.compile(r'\d\d\d\d')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalwindSpeed.append(average)
            total=int(value)
            count=1
    average = total / count
    finalwindSpeed.append(average)

    total = 0
    count = 0
    temp = ''
    for key, value in humidity.items():
        date=key
        dateRegex=re.compile(r'\d\d\d\d')
        mo=dateRegex.search(date)
        if(mo.group()==temp or temp==''):
            temp=mo.group()
            total=total+int(value)
            count=count+1
        else:
            temp=mo.group()
            average=total/count
            finalHumidity.append(average)
            total=int(value)
            count=1
    average = total / count
    finalHumidity.append(average)

print(finalMinTemp)
print(finalMaxTemp)
print(finalwindSpeed)
print(finalHumidity)

