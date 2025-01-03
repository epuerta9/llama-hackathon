# Stage 1: Use the kitchenai base image
FROM epuerta18/kitchenai:latest

# Set up environment variables
ENV PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    PYTHONPATH="/app" \
    PATH="$VENV_PATH/bin:$PATH"


# Install additional dependencies into the existing virtual environment
WORKDIR /tmp

ENV KITCHENAI_MODULE=app:kitchen

COPY requirements.txt requirements.txt

RUN python -m ensurepip && \
    python -m pip install -r requirements.txt

# Set the working directory
WORKDIR /app

# Run migrations and setup tasks using the existing manage.py script


RUN . $VENV_PATH/bin/activate && \
        python kitchenai init


COPY app.py .


# Expose the application port
EXPOSE 8000

# Use the init system defined in the base image
ENTRYPOINT ["/init"]
