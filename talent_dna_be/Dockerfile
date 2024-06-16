# Gunakan image resmi Python sebagai base image
FROM python:3.9

# Set environment variable
ENV PYTHONUNBUFFERED 1

# Buat dan gunakan direktori kerja baru
WORKDIR /code

# Install dependencies
COPY requirements.txt /code
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Salin seluruh kode aplikasi
COPY . .

# Jalankan perintah migrate dan  saat container dimulai
CMD [ "python", "manage.py", "runserver" ]