FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Initialize the database
RUN python -c "from data.db import init_db; init_db()"

EXPOSE 8000

CMD ["python", "web/app.py"]
