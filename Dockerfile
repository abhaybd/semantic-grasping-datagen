ARG GIT_COMMIT
ARG TIMESTAMP

# Use an official Node.js runtime as a parent image
FROM node:20 AS build

# Set the working directory
WORKDIR /app

# Copy the package.json and package-lock.json files
COPY data_annotation/package*.json ./data_annotation/

# Install dependencies
RUN cd data_annotation && npm install

# Copy the rest of the application code
COPY data_annotation ./data_annotation

# Build the React app
ENV REACT_APP_GIT_COMMIT=$GIT_COMMIT
ENV REACT_APP_TIMESTAMP=$TIMESTAMP
RUN cd data_annotation && npm run build

# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY annotation_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r annotation_requirements.txt

COPY setup.py .
COPY README.md .

# Copy the built React app from the previous stage
COPY --from=build /app/data_annotation/build ./data_annotation/build

# Copy the rest of the application code
COPY semantic_grasping_datagen ./semantic_grasping_datagen
COPY categories.txt ./categories.txt
RUN mkdir -p annotations
RUN chmod -R a+rw annotations

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["fastapi", "run", "semantic_grasping_datagen/annotation_server.py", "--port", "8080"]
