# Use the official Python image as a base image
FROM python:3.9

# Set environment variable
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/
RUN pip install -r requirements.txt

# Copy the entire application code
COPY . .

# Run migrations
RUN python manage.py makemigrations
RUN python manage.py migrate

# Expose the port the app runs on
EXPOSE 8000

# Start the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
