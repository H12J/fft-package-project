FROM python:3.9-slim

WORKDIR /app

# Copy the package files
COPY . .

# Install the package
RUN pip install -e .

# Create output directory
RUN mkdir -p output

# Default command: run tests
CMD ["python", "-m", "unittest", "discover", "tests"]