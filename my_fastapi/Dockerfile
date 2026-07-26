FROM python:3.11.15

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--reload"]