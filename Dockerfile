# Step 1: Use an official Python image (Debian-based) to support apt-get
FROM python:3.10

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies (PortAudio for PyAudio)
RUN apt-get update && apt-get install -y portaudio19-dev

# Step 4: Copy project files into the container
COPY . .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Expose the port (Render will override it dynamically)
ENV PORT=5000
EXPOSE 5000

# Step 7: Run the application
CMD ["python", "app.py"]
