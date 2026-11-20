# Base Image Python 3.9 Slim
FROM python:3.9-slim

# Install dependensi sistem untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup User Non-Root untuk keamanan sistem
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Setup direktori kerja aplikasi
WORKDIR /app

# Salin requirements dan instal dependensi
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Salin seluruh file proyek ke dalam container
COPY --chown=user . /app

# Expose port lokal dan jalankan web server Flask
EXPOSE 7860
CMD ["python", "app.py"]
