import os
import json
import base64

with open("temp.json") as f:
    image = json.load(f)
    # decode base64 image string
    image = base64.b64decode(image[0]['image'])

with open("temp.png", "wb") as fh:
    fh.write(image)
