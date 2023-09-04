# First stage: Use the TensorFlow GPU image
FROM tensorflow/tensorflow:1.15.4-gpu-py3 AS tensorflow_stage

# Copy files and install app dependencies.
COPY requirements.txt ./

# Upgrade pip
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Installs
RUN pip install Flask==1.0.3

# Copy the rest of your application files
COPY . .

# Expose the port that our app will be running on
EXPOSE 5000

# Launch the
CMD ["python", "api.py"]
