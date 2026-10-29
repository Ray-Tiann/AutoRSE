ARG AGENT_BASE_IMAGE=autorse-agent:latest
FROM ${AGENT_BASE_IMAGE}
WORKDIR /workspace
COPY requirements.txt /tmp/task-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/task-requirements.txt
COPY . /workspace
RUN chmod +x /workspace/run_baseline.sh && chmod -R a+rwX /workspace
CMD ["/bin/bash"]
