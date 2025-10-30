# Object Detection Microservice

## Overview
A scalable, production-ready object detection microservice built with YOLOv5, FastAPI, and Flask. Detects objects in images with bounding boxes, confidence scores, and detailed metadata.

## Architecture
```
┌─────────────────┐
│   Web Browser   │
│     (User)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  UI Backend     │
│  Flask:8000     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AI Backend     │
│  FastAPI:8001   │
│  YOLOv5 Model   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Shared Storage  │
│   /static/      │
└─────────────────┘
```

### Components
* **UI Backend (Flask)**: Handles web interface, file uploads, and request validation
* **AI Backend (FastAPI)**: Runs YOLOv5 model for object detection
* **Shared Storage**: Stores uploaded images and detection results

## Features
* Real-time object detection using YOLOv5
* Support for 80+ object classes (COCO dataset)
* Bounding box visualization on images
* JSON output with detection metadata
* Containerized deployment with Docker
* Microservices architecture
* Health check endpoints
* Auto-restart on failure

## Technology Stack

### AI Backend
* FastAPI 0.103.1
* YOLOv5n (Ultralytics)
* PyTorch 2.0.1
* TorchVision 0.15.2
* OpenCV-Python-Headless 4.8.0.76
* Pillow 10.0.0
* Pandas 2.0.3
* Uvicorn 0.23.2

### UI Backend
* Flask 2.3.3
* Werkzeug 2.3.7
* Requests 2.31.0
* Pillow 10.0.0

### DevOps
* Docker
* Docker Compose 3.8
* Python 3.9

## Project Structure
```
object-detection-microservice/
├── docker-compose.yaml
├── static/
│   ├── uploads/
│   └── results/
│       ├── image/
│       └── json/
├── ui-backend/
│   ├── Dockerfile
│   ├── Ui-app.py
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html
│   
└── ai-backend/
    ├── Dockerfile
    ├── Ai-app.py
    ├── requirements.txt
   
```

## How to Run

### Prerequisites
1. **Install Docker**:
   * Docker Engine 20.10+
   * Docker Compose 2.0+
   * At least 4GB RAM available
   * At least 5GB free disk space

2. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd object-detection-microservice
   ```

3. **Create Required Directories**:
   ```bash
   mkdir -p static/uploads static/results/image static/results/json
   ```

### Execution

1. **Build and Start Services**:
   ```bash
   docker-compose up --build
   ```

2. **Run in Detached Mode** (optional):
   ```bash
   docker-compose up --build -d
   ```

3. **Access the Application**:
   * **Web Interface**: http://localhost:8000
   * **AI Backend API**: http://localhost:8001
   * **Health Check**: http://localhost:8000/health

4. **Stop Services**:
   ```bash
   docker-compose down
   ```

5. **Stop and Remove Volumes**:
   ```bash
   docker-compose down -v
   ```

## API Documentation

### UI Backend Endpoints

#### POST /upload
Upload an image for object detection.

**Request**:
```bash
curl -X POST http://localhost:8000/upload \
  -F "image=@/path/to/image.jpg"
```

**Response**:
```json
{
  "success": true,
  "image_name": "test.jpg",
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections_count": 3,
  "detections": [
    {
      "object_id": 1,
      "class": "person",
      "confidence": 0.8945,
      "bounding_box": {
        "x_min": 245.32,
        "y_min": 123.45,
        "x_max": 456.78,
        "y_max": 789.12
      },
      "center": {
        "x": 351.05,
        "y": 456.29
      }
    }
  ],
  "result_image": "/static/results/image/result_20231031_123456_test.jpg",
  "result_json": "/static/results/json/result_20231031_123456_test.json",
  "timestamp": "2023-10-31T12:34:56.789"
}
```

#### GET /health
Check service health status.

**Response**:
```json
{
  "status": "healthy",
  "service": "UI Backend",
  "timestamp": "2023-10-31T12:34:56.789",
  "ai_backend": {
    "status": "healthy",
    "model_loaded": true
  }
}
```

### AI Backend Endpoints

#### POST /detect
Directly detect objects in an uploaded image.

**Request**:
```bash
curl -X POST http://localhost:8001/detect \
  -F "file=@/path/to/image.jpg"
```

#### GET /health
Check AI backend health and model status.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": false,
  "results_dir": "/static/results"
}
```

## Steps Followed

### Phase 1: Model Exploration and Setup
* Explored the open-source YOLOv5 model from Ultralytics
* Downloaded and configured YOLOv5n (nano variant) for optimal performance
* Ran basic sanity tests to validate model output
* Verified detection accuracy using COCO dataset classes
* Tested model with various image types and sizes

### Phase 2: AI Backend Development
* Created FastAPI-based backend server for model inference
* Implemented image preprocessing pipeline
* Integrated YOLOv5 model with FastAPI endpoints
* Added bounding box rendering functionality
* Implemented JSON response formatting with detection metadata
* Added error handling and validation

### Phase 3: Testing and Validation
* Conducted unit tests for API endpoints
* Performed integration testing using Postman
* Tested various image formats (JPEG, PNG, WebP, etc.)
* Validated detection accuracy across different object types
* Load tested with concurrent requests
* Verified response time and performance metrics

### Phase 4: UI Backend Development
* Created Flask-based web application
* Designed HTML template for user interface
* Implemented file upload functionality with validation
* Added image type and size validation
* Created request forwarding to AI backend
* Implemented response handling and display logic

### Phase 5: Error Handling and Optimization
* Added comprehensive error handling for file uploads
* Implemented timeout handling for AI backend requests
* Added connection error recovery
* Created validation for image formats and sizes
* Implemented logging for debugging and monitoring
* Added health check endpoints for service monitoring

### Phase 6: Containerization
* Created Dockerfile for AI backend with YOLOv5 dependencies
* Created Dockerfile for UI backend with Flask dependencies
* Configured Docker Compose for service orchestration
* Implemented shared volume for data persistence
* Added health checks for container monitoring
* Configured networking between containers
* Optimized image sizes and build caching

### Phase 7: Documentation and Deployment
* Created comprehensive README with setup instructions
* Documented API endpoints and responses
* Added troubleshooting guide
* Created example requests and responses
* Prepared production deployment guidelines

## Testing

### Manual Testing
1. **Upload Test**:
   ```bash
   curl -X POST http://localhost:8000/upload \
     -F "image=@test.jpg"
   ```

2. **Health Check**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8001/health
   ```

3. **Direct AI Backend Test**:
   ```bash
   curl -X POST http://localhost:8001/detect \
     -F "file=@test.jpg"
   ```

## Docker Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ui-backend
docker-compose logs -f ai-backend
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart ai-backend
```

### Check Service Status
```bash
docker-compose ps
```

### Execute Commands in Container
```bash
# UI Backend
docker-compose exec ui-backend /bin/bash

# AI Backend
docker-compose exec ai-backend /bin/bash
```


### AI Backend Model Loading Issues
* YOLOv5 model downloads on first run (may take several minutes)
* Check AI backend logs:
  ```bash
  docker-compose logs -f ai-backend
  ```
* Ensure stable internet connection for model download

### Connection Issues Between Services
* Verify both services are on the same network:
  ```bash
  docker network inspect object-detection-microservice_detection-network
  ```
* Check service names in docker-compose.yaml match environment variables

### Out of Memory
* Increase Docker memory allocation (Docker Desktop: Settings → Resources → Memory)
* Minimum 4GB RAM recommended

### Image Upload Fails
* Verify file size is under 16MB
* Check file format is supported (PNG, JPG, JPEG, GIF, BMP, WebP)
* Ensure `static/uploads` directory has write permissions

### Detection Takes Too Long
* First request initializes model (slower)
* Subsequent requests should be faster (1-3 seconds)
* Check Docker resource allocation
* Consider using GPU-enabled Docker for faster inference

## References

### Model and Framework
* **YOLOv5**: https://github.com/ultralytics/yolov5
* **Ultralytics Documentation**: https://docs.ultralytics.com/

### Backend Frameworks
* **FastAPI Documentation**: https://fastapi.tiangolo.com/
* **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
* **Flask Documentation**: https://flask.palletsprojects.com/
* **Flask Quickstart**: https://flask.palletsprojects.com/en/latest/quickstart/

### Containerization
* **Docker Documentation**: https://docs.docker.com/
* **Docker Compose**: https://docs.docker.com/compose/
* **Dockerfile Best Practices**: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

### API Testing
* **Postman**: https://www.postman.com/
* **cURL Documentation**: https://curl.se/docs/



## Contact
For questions or support, please open an issue in the repository.
