import base64, requests, cv2, numpy as np
import json
    
input_file = './yolov5/data/images/20.jpg'
data = {}
with open(input_file,'rb') as f:
    data['img']= base64.b64encode(f.read()).decode('utf-8')    
url = "http://localhost:9000/2015-03-31/functions/function/invocations"

response = requests.post(url, json=data)
'''
print('url')
print(response.url)
print('status code')
print(response.status_code)
print('headers')
print(response.headers)
print('encoding')
print(response.encoding)
print('text')
print(response.text)
print('content')
print(response.content)

img_b64 = response.json()['img']
img_bin = base64.b64decode(img_b64)
img_array = np.frombuffer(img_bin,dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
cv2.imwrite('out.png',img)
'''
label = response.json()['label']
print(label)

exit()
