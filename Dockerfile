FROM python:3.11-slim

# Hugging Face Spaces runs the container as user 1000, so the app folder must belong to that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . $HOME/app

EXPOSE 7860
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860", "--headless"]
