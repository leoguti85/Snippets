
# Build the docker image
'''
docker buildx build -f Dockerfile -t frontend-app .
'''

# run the image
'''
docker run -it  -p 8501:8501 frontend-app
'''

