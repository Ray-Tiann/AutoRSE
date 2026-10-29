FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY space/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY --chown=user:user autorse ./autorse
COPY --chown=user:user space ./space
USER user
ENV PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860
CMD ["python", "space/app.py"]
