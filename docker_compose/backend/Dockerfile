FROM python:3.12.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
