# ✅ Use a base image where dlib and face_recognition are already pre-installed
FROM facegen/face_recognition:latest

# Set working directory
WORKDIR /app

# Install Mongo client and dotenv
RUN pip install pymongo python-dotenv

# Copy your app files
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Expose Flask port
EXPOSE 5001

# Start the Flask app
CMD ["flask", "run"]
