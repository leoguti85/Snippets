
# Build the docker image
'''
docker buildx build -f Dockerfile -t llm-app .
'''

# run the image
'''
docker run -it -p 8000:8000 --gpus all llm-app
'''

