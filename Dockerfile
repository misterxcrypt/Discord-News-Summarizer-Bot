# Use the official Python image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies (ensure requirements.txt exists in your project)
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the bot
CMD ["python", "hfbot.py"]
