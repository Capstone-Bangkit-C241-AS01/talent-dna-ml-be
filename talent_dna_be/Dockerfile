# Use the official Python image as a base image
FROM python:3.9

# Set environment variable
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/
RUN apt-get update && apt-get install -y --no-install-recommends netcat-traditional
RUN pip install -r requirements.txt

# Copy the entire application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# run entrypoint.sh
RUN sed -i 's/\r$//g'  /code/entrypoint.sh
RUN chmod +x /code/entrypoint.sh
ENTRYPOINT ["/code/entrypoint.sh"]


# Start the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
