FROM python:3.12-slim
RUN python3 -m pip install --no-cache-dir numpy==1.26.4
WORKDIR /tests
COPY . /tests
CMD ["/bin/bash"]
