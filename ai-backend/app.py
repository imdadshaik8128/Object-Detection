from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
import io
import json
from typing import List, Dict
import logging
from datetime import datetime
import os
import shutil
import pathlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Object Detection AI Backend", version="1.0.0")

BASE_DIR = pathlib.Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR.parent / "static"
# Create results directory
RESULTS_DIR = STATIC_DIR / "results"
IMAGES_DIR = RESULTS_DIR / "image"
JSON_DIR = RESULTS_DIR / "json"

RESULTS_DIR_STR = str(RESULTS_DIR) # Keep for legacy use if needed

os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount static files

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    """Load the YOLOv5 model on startup"""
    global model
    try:
        logger.info("Loading YOLOv5 model...")
        # Load YOLOv5n (nano) model from local directory
        #model = torch.hub.load(f'{BASE_DIR}/models/yolov3', 'yolov5n', source='local')
        model = torch.hub.load('ultralytics/yolov3', 'yolov5n', pretrained=True)
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # IoU threshold
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Backend - Object Detection",
        "model": "YOLOv5n",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "results_dir": RESULTS_DIR
    }

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint to detect objects in an uploaded image
    Uses YOLOv5's built-in rendering for bounding boxes
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with detected objects and result image path
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the image
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Perform detection
        results = model(image)
        
        # Parse results for JSON response
        detections = parse_results(results)
        
        # Generate unique filename for result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(file.filename)[0]
        result_filename_img = f"result_{timestamp}_{base_name}.jpg"
        result_path_img = IMAGES_DIR / result_filename_img 


        img_with_boxes = Image.fromarray(results.render()[0])


        img_with_boxes.save(result_path_img)

        logger.info(f"Detected {len(detections)} objects")
        logger.info(f"Result image saved: {result_path_img}")
        
        response_data = {
            "success": True,
            "image_name": file.filename,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "detections_count": len(detections),
            "detections": detections,
            "result_image": f"/static/results/image/{result_filename_img}",
            #"result_path": str(result_path_img),
            "timestamp": datetime.now().isoformat()
        }
        # Save JSON result to file
        result_filename_json = f"result_{timestamp}_{base_name}.json"
        json_path = JSON_DIR / result_filename_json
        json_path_str = str(json_path)
        with open(json_path_str, 'w') as json_file:
            json.dump(response_data, json_file, indent=4)
        logger.info(f"JSON result saved: {json_path_str}")
        response_data["result_json"] = f"/static/results/json/{result_filename_json}"
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

def parse_results(results) -> List[Dict]:
    """
    Parse YOLO results into structured JSON format
    """
    detections = []

    # Extract predictions
    predictions = results.pandas().xyxy[0]

    # Log raw predictions for debugging
    logger.info(f"Raw predictions:\n{predictions}")

    for idx, row in predictions.iterrows():
        detection = {
            "object_id": idx + 1,
            "class": row['name'],
            "confidence": round(float(row['confidence']), 4),
            "bounding_box": {
                "x_min": round(float(row['xmin']), 2),
                "y_min": round(float(row['ymin']), 2),
                "x_max": round(float(row['xmax']), 2),
                "y_max": round(float(row['ymax']), 2)
            },
            "center": {
                "x": round((float(row['xmin']) + float(row['xmax'])) / 2, 2),
                "y": round((float(row['ymin']) + float(row['ymax'])) / 2, 2)
            }
        }
        detections.append(detection)

    return detections

@app.get("/results/{filename}")
async def get_result_image(filename: str):
    """
    Serve result images
    """
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Result image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)