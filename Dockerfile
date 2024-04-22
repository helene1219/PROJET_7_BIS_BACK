FROM python:3.9-slim as build
EXPOSE 8501
WORKDIR /api/app
COPY /PROJET_7_BIS_BACK/requirements.txt
COPY /PROJET_7_BIS_BACK/src .
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "main.py"]