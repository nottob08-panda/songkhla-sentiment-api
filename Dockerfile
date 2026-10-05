# สำหรับ Hugging Face Spaces (SDK: Docker) หรือโฮสต์ใดก็ได้ที่รัน Docker
FROM python:3.12-slim

# Hugging Face Spaces รันด้วย user id 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user sentiment_core ./sentiment_core
COPY --chown=user api ./api
COPY --chown=user model ./model

USER user
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
