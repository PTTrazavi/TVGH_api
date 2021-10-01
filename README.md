# TVGH_api
## How to use Docker on 201
1. in the directory of docker-compose.yml  
```bash
docker-compose up -d
```
2. go into the running container bash  
```bash
docker exec -it [container ID] bash
```
3. start the flask  
```bash
python3 bin/run.py
```
## the API will work on this URL
https://api.openaifab.com:15001/tvgh_api  
