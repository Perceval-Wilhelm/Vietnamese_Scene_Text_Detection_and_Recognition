# Use the PaddlePaddle Docker image
FROM paddlepaddle/paddle:2.6.0

# Set the working directory in the container
WORKDIR /app

# Install the necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
ENTRYPOINT ["streamlit", "run"]

# Streamlit app filename
CMD ["app.py"]