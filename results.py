import requests
import json

# set the API endpoint
url = 'http://localhost:5000/result'

# define the input data as a dictionary
data = {'data': "is this looks like spam"}

# convert the data to a JSON string
json_data = json.dumps(data)

# set the headers to specify the content type
headers = {'Content-Type': 'application/json'}

# make the request to the API
response = requests.post(url, data=json_data, headers=headers)

# print the response from the API
if response.status_code == 200:

    # check if the response is valid JSON
    try:
        response_json = response.json()
        print(response_json)

    except json.decoder.JSONDecodeError:
        print("Invalid JSON response.")

else:
    print("Error:", response.status_code)
