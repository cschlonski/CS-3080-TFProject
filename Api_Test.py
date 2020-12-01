'''
LIST OF ITEMS YOU CAN GET FROM THE API:
It is setup as a dictionary so you will need to interect with it that way. The outside for loop with give overall data(anything inline with the date line below) and the inside for loop will give data from the
hourly section. I am guessing there is a better way to get the hourly stuff to work but I got tired of messing with it.
 {
                "date": "2013-04-21",
                "astronomy": [
                    {
                        "sunrise": "06:14 AM",
                        "sunset": "07:43 PM",
                        "moonrise": "03:33 PM",
                        "moonset": "03:36 AM",
                        "moon_phase": "Waxing Gibbous",
                        "moon_illumination": "76"
                    }
                ],
                "maxtempC": "10",
                "maxtempF": "50",
                "mintempC": "-1",
                "mintempF": "30",
                "avgtempC": "4",
                "avgtempF": "40",
                "totalSnow_cm": "2.0",
                "sunHour": "11.6",
                "uvIndex": "2",
                "hourly": [
                    {
                        "time": "0",
                        "tempC": "0",
                        "tempF": "33",
                        "windspeedMiles": "5",
                        "windspeedKmph": "8",
                        "winddirDegree": "224",
                        "winddir16Point": "SW",
                        "weatherCode": "113",
                        "weatherIconUrl": [
                            {
                                "value": "http://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0008_clear_sky_night.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Clear"
                            }
                        ],
                        "precipMM": "0.0",
                        "precipInches": "0.0",
                        "humidity": "77",
                        "visibility": "10",
                        "visibilityMiles": "6",
                        "pressure": "1018",
                        "pressureInches": "31",
                        "cloudcover": "22",
                        "HeatIndexC": "0",
                        "HeatIndexF": "33",
                        "DewPointC": "-3",
                        "DewPointF": "26",
                        "WindChillC": "-2",
                        "WindChillF": "28",
                        "WindGustMiles": "11",
                        "WindGustKmph": "17",
                        "FeelsLikeC": "-2",
                        "FeelsLikeF": "28",
                        "uvIndex": "1"
                    },
                    {
                        "time": "300",
                        "tempC": "-1",
                        "tempF": "30",
                        "windspeedMiles": "6",
                        "windspeedKmph": "10",
                        "winddirDegree": "267",
                        "winddir16Point": "W",
                        "weatherCode": "116",
                        "weatherIconUrl": [
                            {
                                "value": "http://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0004_black_low_cloud.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Partly cloudy"
                            }
                        ],
                        "precipMM": "0.0",
                        "precipInches": "0.0",
                        "humidity": "80",
                        "visibility": "10",
                        "visibilityMiles": "6",
                        "pressure": "1017",
                        "pressureInches": "31",
                        "cloudcover": "44",
                        "HeatIndexC": "-1",
                        "HeatIndexF": "30",
                        "DewPointC": "-4",
                        "DewPointF": "25",
                        "WindChillC": "-5",
                        "WindChillF": "24",
                        "WindGustMiles": "13",
                        "WindGustKmph": "20",
                        "FeelsLikeC": "-5",
                        "FeelsLikeF": "24",
                        "uvIndex": "1"
                    },
                    {
                        "time": "600",
                        "tempC": "-1",
                        "tempF": "30",
                        "windspeedMiles": "7",
                        "windspeedKmph": "12",
                        "winddirDegree": "288",
                        "winddir16Point": "WNW",
                        "weatherCode": "113",
                        "weatherIconUrl": [
                            {
                                "value": "http://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0001_sunny.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Sunny"
                            }
                        ],
                        "precipMM": "0.0",
                        "precipInches": "0.0",
                        "humidity": "79",
                        "visibility": "10",
                        "visibilityMiles": "6",
                        "pressure": "1019",
                        "pressureInches": "31",
                        "cloudcover": "9",
                        "HeatIndexC": "-1",
                        "HeatIndexF": "30",
                        "DewPointC": "-4",
                        "DewPointF": "24",
                        "WindChillC": "-5",
                        "WindChillF": "23",
                        "WindGustMiles": "15",
                        "WindGustKmph": "24",
                        "FeelsLikeC": "-5",
                        "FeelsLikeF": "23",
                        "uvIndex": "2"
                    },
                    {
                        "time": "900",
                        "tempC": "5",
                        "tempF": "41",
                        "windspeedMiles": "6",
                        "windspeedKmph": "9",
                        "winddirDegree": "285",
                        "winddir16Point": "WNW",
                        "weatherCode": "113",
                        "weatherIconUrl": [
                            {
                                "value": "http://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0001_sunny.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Sunny"
                            }
                        ],
                        "precipMM": "0.0",
                        "precipInches": "0.0",
                        "humidity": "53",
                        "visibility": "10",
                        "visibilityMiles": "6",
                        "pressure": "1017",
                        "pressureInches": "31",
                        "cloudcover": "5",
                        "HeatIndexC": "5",
                        "HeatIndexF": "41",
                        "DewPointC": "-4",
                        "DewPointF": "25",
                        "WindChillC": "3",
                        "WindChillF": "37",
                        "WindGustMiles": "6",
                        "WindGustKmph": "10",
                        "FeelsLikeC": "3",
                        "FeelsLikeF": "37",
                        "uvIndex": "3"
                    },
                    {
                        "time": "1200",
                        "tempC": "10",
                        "tempF": "49",
                        "windspeedMiles": "4",
                        "windspeedKmph": "6",
                        "winddirDegree": "157",
                        "winddir16Point": "SSE",
                        "weatherCode": "113",
                        "weatherIconUrl": [
                            {
                                "value": "http://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0001_sunny.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Sunny"
                            }
                        ],
                        "precipMM": "0.0",
                        "precipInches": "0.0",
                        "humidity": "40",
                        "visibility": "10",
                        "visibilityMiles": "6",
                        "pressure": "1016",
                        "pressureInches": "30",
                        "cloudcover": "7",
                        "HeatIndexC": "10",
                        "HeatIndexF": "50",
                        "DewPointC": "-3",
                        "DewPointF": "27",
                        "WindChillC": "9",
                        "WindChillF": "49",
                        "WindGustMiles": "4",
                        "WindGustKmph": "7",
                        "FeelsLikeC": "9",
                        "FeelsLikeF": "49",
                        "uvIndex": "3"
                    },
                    {
                        "time": "1500",
                        "tempC": "10",
                        "tempF": "50",
                        "windspeedMiles": "6",
                        "windspeedKmph": "9",
                        "winddirDegree": "214",
                        "winddir16Point": "SW",
                        "weatherCode": "176",
                        "weatherIconUrl": [
                            {
                                "value": "http://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0009_light_rain_showers.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Patchy rain possible"
                            }
                        ],
                        "precipMM": "0.5",
                        "precipInches": "0.0",
                        "humidity": "42",
                        "visibility": "9",
                        "visibilityMiles": "5",
                        "pressure": "1014",
                        "pressureInches": "30",
                        "cloudcover": "46",
                        "HeatIndexC": "10",
                        "HeatIndexF": "50",
                        "DewPointC": "-2",
                        "DewPointF": "28",
                        "WindChillC": "9",
                        "WindChillF": "48",
                        "WindGustMiles": "7",
                        "WindGustKmph": "11",
                        "FeelsLikeC": "9",
                        "FeelsLikeF": "48",
                        "uvIndex": "3"
                    },
                    {
                        "time": "1800",
                        "tempC": "8",
                        "tempF": "47",
                        "windspeedMiles": "5",
                        "windspeedKmph": "8",
                        "winddirDegree": "24",
                        "winddir16Point": "NNE",
                        "weatherCode": "200",
                        "weatherIconUrl": [
                            {
                                "value": "http://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0032_thundery_showers_night.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Thundery outbreaks possible"
                            }
                        ],
                        "precipMM": "0.0",
                        "precipInches": "0.0",
                        "humidity": "55",
                        "visibility": "9",
                        "visibilityMiles": "5",
                        "pressure": "1014",
                        "pressureInches": "30",
                        "cloudcover": "45",
                        "HeatIndexC": "8",
                        "HeatIndexF": "47",
                        "DewPointC": "-0",
                        "DewPointF": "32",
                        "WindChillC": "7",
                        "WindChillF": "45",
                        "WindGustMiles": "6",
                        "WindGustKmph": "10",
                        "FeelsLikeC": "7",
                        "FeelsLikeF": "45",
                        "uvIndex": "1"
                    },
                    {
                        "time": "2100",
                        "tempC": "3",
                        "tempF": "38",
                        "windspeedMiles": "2",
                        "windspeedKmph": "4",
                        "winddirDegree": "149",
                        "winddir16Point": "SSE",
                        "weatherCode": "371",
                        "weatherIconUrl": [
                            {
                                "value": "http://cdn.worldweatheronline.com/images/wsymbols01_png_64/wsymbol_0028_heavy_snow_showers_night.png"
                            }
                        ],
                        "weatherDesc": [
                            {
                                "value": "Moderate or heavy snow showers"
                            }
                        ],
                        "precipMM": "2.4",
                        "precipInches": "0.1",
                        "humidity": "84",
                        "visibility": "7",
                        "visibilityMiles": "4",
                        "pressure": "1015",
                        "pressureInches": "30",
                        "cloudcover": "49",
                        "HeatIndexC": "3",
                        "HeatIndexF": "38",
                        "DewPointC": "1",
                        "DewPointF": "34",
                        "WindChillC": "3",
                        "WindChillF": "37",
                        "WindGustMiles": "5",
                        "WindGustKmph": "8",
                        "FeelsLikeC": "3",
                        "FeelsLikeF": "37",
                        "uvIndex": "1"
                    }
                ]
            },
'''



def api_search(code,start,end):
    url="http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=6230647e6745477995c00355202410&q="+code+"&format=json&date="+start+"&enddate="+end
    return url

import json
import urllib.request

code=input('Enter a zip code:')
start=input('Enter a start date in the format yyyy-mm-dd:')
end=input('Enter an end date in the format yyyy-mm-dd:')

url = api_search(code,start,end)

obj = urllib.request.urlopen(url)

rawdata = json.load(obj)

data = rawdata['data']

print("\n")

for item in data['weather']:
    i=0
    print('Date:'+ item['date'])
    print('\tMixtemp:'+item['mintempF']+'F')
    print('\tMaxtemp:'+item['maxtempF']+'F')
    for item2 in item['hourly']:
        test=item['hourly']
        test2 = test[i]
        print('\tTime:'+test2['time'])
        print('\t\tTemp:'+test2['tempF']+'F')
        print('\t\tTemp:' + test2['tempF'] + 'F\n')
        i=i+1

