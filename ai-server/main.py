import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect
import json
import os
import time
from collections import deque
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
import tempfile
import datetime
from urllib.parse import urlparse,unquote
import ffmpeg
import torch



app = FastAPI()
clients = []

temp = []

# Azure Blob Storage 정보 설정
AZURE_STORAGE_ACCOUNT_NAME = 'mozitstorage'
AZURE_CONTAINER_NAME = 'mozit-container'
SAS_TOKEN = os.getenv("SAS_TOKEN")
if SAS_TOKEN is None:
    raise ValueError("SAS_TOKEN environment variable is not set")

# BlobServiceClient 생성
blob_service_client = BlobServiceClient(
    f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net?{SAS_TOKEN}"
)

# 허용할 출처
origins = [
    "https://ambitious-grass-00e12ba00.4.azurestaticapps.net",  # React 애플리케이션의 주소
    # 추가 출처를 여기서 설정 가능
]

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 출처
    allow_credentials=True,   # 쿠키와 인증 정보를 포함한 요청을 허용
    allow_methods=["*"],      # 모든 HTTP 메서드 허용
    allow_headers=["*"],      # 모든 헤더 허용
)


# Load YOLO models
model1_path = "./face_yolo11n.onnx"  # First YOLO model
model2_path = "./personal_info_yolo11n.onnx"  # Second YOLO model
model3_path = "./H_model_n.pt"  # Third YOLO model

model1 = YOLO(model1_path)
model2 = YOLO(model2_path)
model3 = YOLO(model3_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1.to(device)
# model2.to(device)
# model3.to(device)

# Object tracking settings
next_object_id = 0
object_tracks = {}  # Stores object_id and their last known positions and bounding boxes
max_distance = 50  # Maximum distance to consider the same object
min_iou = 0.3  # Minimum IoU to consider the same object

# Calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def apply_mosaic(frame, x, y, w, h, intensity=15, size=0, hardness=1):
    """Apply mosaic with specified intensity and hardness to the bounding box."""
    # Ensure intensity is at least 1
    intensity = max(1, intensity)
    
    # Avoid processing very small boxes
    if w <= 1 or h <= 1:
        return frame

    # Extract region of interest (ROI)
    roi = frame[y:y+h, x:x+w]

    # Calculate block size based on React's parameters
    block_size = max(size / 4, 4) * (intensity / 100)  # React의 방식으로 조정

    # Calculate small dimensions for mosaic
    small_w = max(1, int(w / block_size))  # Ensure it's at least 1
    small_h = max(1, int(h / block_size))  # Ensure it's at least 1

    # Apply resizing for mosaic
    try:
        roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"Error during resizing: {e}")
        return frame  # Return original frame if resizing fails

    # Blend the mosaic ROI back into the frame based on hardness
    alpha = max(0, min(1, hardness))  # Clamp hardness between 0 and 1
    mosaic_roi = roi.astype(np.float32)
    original_roi = frame[y:y+h, x:x+w].astype(np.float32)
    blended_roi = cv2.addWeighted(mosaic_roi, alpha, original_roi, 1 - alpha, 0)

    # Replace the original region with the blended ROI
    frame[y:y+h, x:x+w] = blended_roi.astype(np.uint8)

    return frame

def process_video(video_path):
    video_path = urlparse(video_path)
    video_path = video_path.path.lstrip('/')  # 앞쪽 '/' 제거
    
    # 컨테이너 이름이 포함되어 있으면 제거
    if video_path.startswith(AZURE_CONTAINER_NAME + '/'):
        video_path = video_path[len(AZURE_CONTAINER_NAME) + 1:]
        
    temp_file_path = None  # 임시 파일 경로 저장 변수
    try:
        # BlobClient 생성 (SAS URL로 접근)
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=video_path)

        # 임시 파일 경로 생성
        temp_file_path = os.path.join(tempfile.gettempdir(), os.path.basename(video_path))

        try:
            # Azure Blob Storage에서 파일 다운로드
            download_blob = blob_client.download_blob()
            if download_blob:
                with open(temp_file_path, "wb") as download_file:
                    download_file.write(download_blob.readall())
            else:
                 raise ValueError("process_video_Downloaded blob is empty")
        except Exception as e:
            print(f"[ERROR] process_video Error downloading video file: {e}")
            return

        # 모델 로드 (각각의 YOLO 모델)
        models = [model1, model2, model3]  # 사전에 로드된 3개의 모델

        # 비디오 입력
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            print("[ERROR] Failed to open video file.")
            return

        # 모든 감지 결과 저장
        all_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            combined_boxes = []
            combined_classes = []
            combined_scores = []
            combined_ids = []

            # 각 모델로부터 결과 추출
            for model_idx, model in enumerate(models):
                try:
                    # 추적 결과 가져오기
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device="cpu")
                    if results is None or not hasattr(results[0], 'boxes') or results[0].boxes is None:
                        print(f"[DEBUG] No valid results for model {model_idx} in this frame.")
                        continue

                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.numpy()
                        classes = results[0].boxes.cls.numpy()
                        scores = results[0].boxes.conf.numpy()
                        ids = results[0].boxes.id.numpy()
                        
                        class_names = [model.names[int(cls)] for cls in classes]
                        combined_boxes.append(boxes)
                        combined_classes.extend(class_names)
                        combined_scores.append(scores)
                        combined_ids.append(ids)
                    else:
                        print(f"[DEBUG] No objects detected by model {model_idx} in this frame.")
                except Exception as e:
                    print(f"[ERROR] Error processing model {model_idx}: {e}")

            # 결과를 통합
            if len(combined_boxes) > 0:
                combined_boxes = np.vstack(combined_boxes)
                combined_classes = combined_classes
                combined_scores = np.hstack(combined_scores)
                combined_ids = np.hstack(combined_ids)
            else:
                combined_boxes = np.empty((0, 4))
                combined_classes = []
                combined_scores = np.empty((0,))
                combined_ids = np.empty((0,))

            # 감지 결과가 없는 경우 현재 프레임 건너뜀
            if len(combined_boxes) == 0:
                frame_detections = []
                all_detections.append({"frame": len(all_detections) + 1, "detections": frame_detections})
                # out.write(frame)
                continue

            frame_detections = []

            for box, class_name, score, object_id in zip(combined_boxes, combined_classes, combined_scores, combined_ids):
                x_min, y_min, x_max, y_max = map(int, box[:4])

                # JSON 구조에 추가
                frame_detections.append({
                    "objectId": int(object_id),  # 추적 ID
                    "className": class_name,
                    "x": float(x_min),
                    "y": float(y_min),
                    "width": float(x_max - x_min),
                    "height": float(y_max - y_min),
                    "confidence": float(score)
                })

            # 감지 결과를 저장
            all_detections.append({"frame": len(all_detections) + 1, "detections": frame_detections})

        cap.release()

    except Exception as e:
        print(f"[ERROR] An error occurred during video processing: {e}")

    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"[INFO] Temporary file {temp_file_path} has been deleted.")
            except Exception as e:
                print(f"[ERROR] Error deleting temporary file {temp_file_path}: {e}")

    global temp
    temp=all_detections.copy()
    return all_detections



def process_image(image_path):
    # 이미지 파일 확인
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file at {image_path} does not exist.")
    
    # 모델 로드 (각각의 YOLO 모델)
    models = [model1, model2, model3]  # 사전에 로드된 3개의 모델
    
    # 이미지 입력
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read the image at {image_path}.")
    
    base_name = os.path.basename(image_path)
    output_path = os.path.join(os.path.dirname(image_path), "processed_" + base_name)
    
    combined_boxes = []
    combined_classes = []
    combined_scores = []
    combined_ids = []

    # 각 모델로부터 결과 추출
    for model_idx, model in enumerate(models):
        try:
            # 추적 결과 가져오기
            results = model.track(image, persist=True, tracker="bytetrack.yaml")
            if results is None or not hasattr(results[0], 'boxes') or results[0].boxes is None:
                print(f"[DEBUG] No valid results for model {model_idx}.")
                continue
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.numpy()
                classes = results[0].boxes.cls.numpy()
                scores = results[0].boxes.conf.numpy()
                ids = results[0].boxes.id.numpy()
                
                class_names = [model.names[int(cls)] for cls in classes]
                combined_boxes.append(boxes)
                combined_classes.extend(class_names)
                combined_scores.append(scores)
                combined_ids.append(ids)
            else:
                print(f"[DEBUG] No objects detected by model {model_idx}.")
        except Exception as e:
            print(f"[ERROR] Error processing model {model_idx}: {e}")
    
    # 결과를 통합
    if len(combined_boxes) > 0:
        combined_boxes = np.vstack(combined_boxes)
        combined_classes = combined_classes
        combined_scores = np.hstack(combined_scores)
        combined_ids = np.hstack(combined_ids)
    else:
        combined_boxes = np.empty((0, 4))
        combined_classes = []
        combined_scores = np.empty((0,))
        combined_ids = np.empty((0,))
    
    # 감지 결과가 없는 경우 원본 이미지 그대로 반환
    if len(combined_boxes) == 0:
        cv2.imwrite(output_path, image)
        print(f"Processed image saved to {output_path}")
        return []

    frame_detections = []

    # 감지된 객체들에 대해 모자이크 적용 및 바운딩 박스 그리기
    for box, class_name, score, object_id in zip(combined_boxes, combined_classes, combined_scores, combined_ids):
        x_min, y_min, x_max, y_max = map(int, box[:4])
        w, h = x_max - x_min, y_max - y_min

        # JSON 구조에 추가
        frame_detections.append({
            "objectId": int(object_id),  # 추적 ID
            "className": class_name,
            "x": float(x_min),
            "y": float(y_min),
            "width": float(x_max - x_min),
            "height": float(y_max - y_min),
            "confidence": float(score)
        })

        # 모자이크 적용
        image = apply_mosaic(image, x_min, y_min, w, h)

    # 결과를 이미지로 저장
    cv2.imwrite(output_path, image)
    
    # 감지 결과를 JSON 파일로 저장
    json_path = output_path.replace(".jpg", ".json").replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(frame_detections, f, indent=4)

    print(f"Processed image saved to {output_path}")
    print(f"Detection results saved to {json_path}")
    
    return frame_detections

def input_editor(video_path, output_path, parameters, data, video_title):
    # 비디오 파일 확인
    temp_file_path = None  # 임시 파일 경로 저장 변수
    video_path = urlparse(video_path)
    video_path = video_path.path.lstrip('/')  # 앞쪽 '/' 제거
    
    # 컨테이너 이름이 포함되어 있으면 제거
    if video_path.startswith(AZURE_CONTAINER_NAME + '/'):
        video_path = video_path[len(AZURE_CONTAINER_NAME) + 1:]
    
    # BlobClient 생성 (SAS URL로 접근)
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=video_path)

    # 임시 파일 경로 생성
    temp_file_path = os.path.join(tempfile.gettempdir(), os.path.basename(video_path))

    try:
        with open(temp_file_path, "wb") as download_file:
            download_blob = blob_client.download_blob()
            download_file.write(download_blob.readall())
    except Exception as e:
        print(f"[ERROR] Error downloading video file: {e}")
        return
    
    cap = cv2.VideoCapture(temp_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    temp_output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 해당 프레임의 탐지 정보 가져오기
        for frame_data in data:
            if frame_data["frame"] == frame_idx:
                for obj in frame_data["detections"]:
                    obj_id = obj['objectId']
                    class_name = obj['className']
                    
                    if class_name in parameters['harmful']['checklist']:
                        x, y, w, h = map(int, [obj["x"], obj["y"], obj["width"], obj["height"]])
                        intensity = parameters['harmful']['intensity']
                        size = parameters['harmful']['size']
                        
                        w = w + size
                        h = h + size
                        
                        x = max(0, x - size // 2)
                        y = max(0, y - size // 2)
                        w = min(width - x, w)
                        h = min(height - y, h)
                        
                        frame = apply_mosaic(frame, x, y, w, h, intensity, size)
                    elif class_name in parameters['privacy']['checklist']:
                        x, y, w, h = map(int, [obj["x"], obj["y"], obj["width"], obj["height"]])
                        intensity = parameters['privacy']['intensity']
                        size = parameters['privacy']['size']
                        
                        w = w + size
                        h = h + size
                        
                        x = max(0, x - size // 2)
                        y = max(0, y - size // 2)
                        w = min(width - x, w)
                        h = min(height - y, h)
                        
                        frame = apply_mosaic(frame, x, y, w, h, intensity, size)
                    elif class_name =="face":
                        if obj_id in parameters['person']['checklist']:
                            x, y, w, h = map(int, [obj["x"], obj["y"], obj["width"], obj["height"]])
                            intensity = parameters['person']['intensity']
                            size = parameters['person']['size']
                            
                            w = w + size
                            h = h + size
                            
                            x = max(0, x - size // 2)
                            y = max(0, y - size // 2)
                            w = min(width - x, w)
                            h = min(height - y, h)
                            
                            frame = apply_mosaic(frame, x, y, w, h, intensity, size)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    # 영상에 오디오 결합
    temp_merged_path = os.path.join(tempfile.gettempdir(), "tmp_merged.mp4")
    try:
        # FFmpeg 명령어 실행 (모자이크된 비디오는 그대로 사용, 오디오는 원본에서 가져옴)
        video_input = ffmpeg.input(temp_output_path)  # 모자이크된 영상 (비디오만)
        audio_input = ffmpeg.input(temp_file_path)   # 원본 영상 (오디오만)
        
        # 비디오와 오디오 결합
        merge=ffmpeg.output(audio_input.audio, video_input.video, temp_merged_path)
        merge.run()
            
        print(f"✅ Successfully merged audio into {temp_merged_path}")

    except Exception as e:
        print(f"[ERROR] FFmpeg merging failed: {e}")
        return 
    
    
    blobname= f"uploaded-videos/{video_title}.mp4"
    
    # BlockBlobClient 생성
    blob_client_download = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blobname)
    try:
        # Blob 업로드 (temp_merged_path 파일을 업로드)
        with open(temp_merged_path, "rb") as temp_data:
            blob_client_download.upload_blob(temp_data, overwrite=True)
        print(f"Successfully uploaded {blobname} to Azure Blob Storage.")
        
    except Exception as e:
        print(f"[ERROR] Error uploading video to Azure: {e}")
    
    # 임시 파일 삭제
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)        
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)    
        if os.path.exists(temp_merged_path):
            os.remove(temp_merged_path) 
        print("✅ Successfully deleted temporary files.")
    except Exception as e:
        print(f"[ERROR] Error deleting temporary files: {e}")
    
    
    # 업로드된 Blob URL 반환
    return blob_client_download.url

class VideoPathRequest(BaseModel):
    video_path: str
    output_path: str
    video_title: str

class InputEditorRequest(BaseModel):
    harmful_intensity: int
    harmful_size: int
    harmful_checklist: str
    
    privacy_intensity: int
    privacy_size: int
    privacy_checklist: str
    
    person_intensity: int
    person_size: int
    person_checklist: str
    

@app.post("/input_editor")
async def input_editor_endpoint(request: InputEditorRequest,   #모자이크 정보
                                path_request: VideoPathRequest): #비디오 경로
    harmful_intensity = request.harmful_intensity
    harmful_size = request.harmful_size
    harmful_checklist = request.harmful_checklist
    harmful_checklist = [x.strip() for x in harmful_checklist.split(",")] if harmful_checklist.strip() else []
    
    privacy_intensity = request.privacy_intensity
    privacy_size = request.privacy_size
    privacy_checklist = request.privacy_checklist
    privacy_checklist = [x.strip() for x in privacy_checklist.split(",")] if privacy_checklist.strip() else []
    
    person_intensity = request.person_intensity
    person_size = request.person_size
    person_checklist = request.person_checklist
    person_checklist = [int(x.strip()) for x in person_checklist.split(",")] if person_checklist.strip() else []
    
    parameters = {'harmful' : {'intensity' : harmful_intensity,
                                'size' : harmful_size,
                                'checklist' : harmful_checklist},
                  'privacy' : {'intensity' : privacy_intensity,
                                'size' : privacy_size,
                                'checklist' : privacy_checklist},
                  'person' : {'intensity' : person_intensity,
                                'size' : person_size,
                                'checklist' : person_checklist},
                  }
    
    video_path = path_request.video_path
    output_path = path_request.output_path
    video_title = path_request.video_title
    
    data = temp
    
    # 영상 처리 함수 호출
    try:
        output = input_editor(video_path, output_path, parameters, data, video_title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    return output


class VideoPathRequest2(BaseModel):
    video_path: str

@app.post("/capture_thumbnail")
async def capture_thumbnail(request: VideoPathRequest2):  # 풀 경로를 직접 받음
    video_url = request.video_path
    print(f"Video URL: {video_url}")
    
    # video_url에서 블롭 경로 추출 (SAS 토큰 제거)
    video_url = urlparse(video_url)
    video_url = video_url.path.lstrip('/')  # 앞쪽 '/' 제거

    # 컨테이너 이름이 포함되어 있으면 제거
    if video_url.startswith(AZURE_CONTAINER_NAME + '/'):
        video_url = video_url[len(AZURE_CONTAINER_NAME) + 1:]

    print(f"Extracted blob path: {video_url}")  # 디버깅용

    # BlobClient 생성 (SAS URL로 접근)
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=video_url)
    print(f"Blob URL: {blob_client.url}")
    if not blob_client.exists():
        print("Blob does not exist")    
    # 임시 파일 경로 생성
    temp_file_path = os.path.join(tempfile.gettempdir(), os.path.basename(video_url))
    print(f"temp_file_path: {temp_file_path}")
    try:
        # Azure Blob Storage에서 파일 다운로드
        try:
            download_blob = blob_client.download_blob()
            with open(temp_file_path, "wb") as download_file:
                download_file.write(download_blob.readall())
        except Exception as e:
            print(f"Download failed: {e}")

        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(temp_file_path)

        # 비디오 파일 열기 확인
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to access the video.")
        
        # 첫 번째 프레임 읽기
        ret, frame = cap.read()
        if ret:

            # 썸네일 이미지 파일 이름 설정
            blob_name = f"thumbnail/{os.path.basename(video_url).split('.')[0]}_thumbnail.jpg"

            # 썸네일을 Azure Blob Storage에 저장
            _, buffer = cv2.imencode('.jpg', frame)  # 프레임을 JPEG 형식으로 인코딩
            blob_client_thumbnail = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)

            # Blob에 이미지 데이터 업로드
            blob_client_thumbnail.upload_blob(buffer.tobytes(), overwrite=True)

            # 업로드된 파일의 URL을 자동으로 가져오기
            thumbnail_url = blob_client_thumbnail.url

            return {
                "message": "Thumbnail saved successfully.",
                "thumbnail_url": thumbnail_url  # 저장된 썸네일 URL 반환
            }
        else:
            raise HTTPException(status_code=400, detail="첫 프레임을 캡처할 수 없습니다.")

    finally:
        # 비디오 캡처 객체 해제
        if cap is not None:
            cap.release()

        # 임시 비디오 파일 삭제
        if os.path.isfile(temp_file_path):
            os.remove(temp_file_path)
            

class VideoPathRequest1(BaseModel):
    video_path: str
    output_path: str

@app.post("/process-video/")
async def process_video_endpoint(request: VideoPathRequest1):
    print(f"Received request: video_path={request.video_path}, output_path={request.output_path}")
    video_path = request.video_path
    output_path = request.output_path
    
    # 영상 처리 함수 호출
    try:
        data = process_video(video_path)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    
@app.post("/process-image/")
async def process_image_endpoint(image_path: str = Query(...)):
    # 입력된 영상 경로 확인
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"The image file at {image_path} does not exist.")

    # 영상 처리 함수 호출
    try:
        data = process_image(image_path)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
    
@app.websocket("/process-screen/")
async def process_screen_endpoint(websocket:WebSocket):
    await websocket.accept()
    clients.append(websocket)
    
    settings = {}
    
    # 모델 로드 (각각의 YOLO 모델)
    models = [model1, model2, model3]  # 사전에 로드된 3개의 모델
    
    # delay 설정
    delay_seconds=3
    
    # 프레임을 저장할 큐를 생성 (길이는 3초에 해당하는 프레임 수)
    frame_queue = deque(maxlen=30 * delay_seconds)  # 큐 크기: 3초 동안의 프레임 수
    time_queue = deque(maxlen=30 * delay_seconds)  # 프레임 캡처 시간을 저장하는 큐
    
    # 영상 처리 함수 호출
    try:
        while True:
            data = await websocket.receive()
            # 설정 데이터 수신
            if isinstance(data, dict) and "text" in data: 
                try:
                    settings = json.loads(data["text"])
                    settings = json.loads(settings["text"])
                    # 모자이크 변수
                    harmful_intensity = settings["harmful"]["intensity"]
                    harmful_size = settings["harmful"]["size"]
                    harmful_checklist = settings["harmful"]["checklist"]
                    harmful_checklist = [x.strip() for x in harmful_checklist.split(",")] if harmful_checklist.strip() else []
                    
                    privacy_intensity = settings["privacy"]["intensity"]
                    privacy_size = settings["privacy"]["size"]
                    privacy_checklist = settings["privacy"]["checklist"]
                    privacy_checklist = [x.strip() for x in privacy_checklist.split(",")] if privacy_checklist.strip() else []
                    
                    person_intensity = settings["person"]["intensity"]
                    person_size = settings["person"]["size"]
                    person_checklist = settings["person"]["checklist"]
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[WARNING] JSON 데이터 처리 중 오류 발생: {e}")
                    continue
            # 프레임 데이터 (바이너리) 수신
            elif isinstance(data, dict) and "bytes" in data:
                np_data = np.frombuffer(data["bytes"], np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)       
            
                # 현재 시간 저장
                current_time = time.time()

                # 각 모델로부터 결과 추출
                combined_boxes = []
                combined_classes = []

                for model_idx, model in enumerate(models):
                    try:
                        # 추적 결과 가져오기
                        results = model.track(frame, persist=True, tracker="bytetrack.yaml", device="cpu")
                        if results is None or not hasattr(results[0], 'boxes') or results[0].boxes is None:
                            print(f"[DEBUG] No valid results for model {model_idx}.")
                            continue
                        
                        if len(results[0].boxes) > 0:
                            boxes = results[0].boxes.xyxy.numpy()
                            classes = results[0].boxes.cls.numpy()
                            
                            class_names = [model.names[int(cls)] for cls in classes]
                            combined_boxes.append(boxes)
                            combined_classes.extend(class_names)
                        else:
                            print(f"[DEBUG] No objects detected by model {model_idx}.")
                    except Exception as e:
                        print(f"[ERROR] Error processing model {model_idx}: {e}")

                # 결과를 통합
                if len(combined_boxes) > 0:
                    combined_boxes = np.vstack(combined_boxes)
                    combined_classes = combined_classes
                else:
                    combined_boxes = np.empty((0, 4))
                    combined_classes = []

                frame_copy = frame.copy()
                height, width, _ = frame.shape
            
                for box, class_name in zip(combined_boxes, combined_classes):
                    x_min, y_min, x_max, y_max = map(int, box[:4])
                    w, h = max(x_max - x_min, 1), max(y_max - y_min, 1)
                    x, y = max(x_min, 0), max(y_min, 0)
                    # 모자이크 적용
                    # 해당 프레임의 탐지 정보 가져오기
                            
                    if class_name in harmful_checklist:
                        intensity = harmful_intensity
                        size = harmful_size
                        
                        w = w + size
                        h = h + size
                        
                        x = max(0, x - size // 2)
                        y = max(0, y - size // 2)
                        w = min(width - x, w)
                        h = min(height - y, h)
                        
                        frame_copy = apply_mosaic(frame_copy, x, y, w, h, intensity, size)
                    elif class_name in privacy_checklist:
                        intensity = privacy_intensity
                        size = privacy_size
                        
                        w = w + size
                        h = h + size
                        
                        x = max(0, x - size // 2)
                        y = max(0, y - size // 2)
                        w = min(width - x, w)
                        h = min(height - y, h)
                        
                        frame_copy = apply_mosaic(frame_copy, x, y, w, h, intensity, size)
                    elif class_name == "face":
                        if person_checklist:
                            intensity = person_intensity
                            size = person_size
                            
                            w = w + size
                            h = h + size
                            
                            x = max(0, x - size // 2)
                            y = max(0, y - size // 2)
                            w = min(width - x, w)
                            h = min(height - y, h)
                        
                            frame_copy = apply_mosaic(frame_copy, x, y, w, h, intensity, size)
                    

                # 프레임과 시간을 큐에 저장
                frame_queue.append(frame_copy)
                time_queue.append(current_time)

                # 3초가 지난 프레임을 출력
                if len(time_queue) > 0 and current_time - time_queue[0] >= delay_seconds:
                    frame_to_show = frame_queue.popleft()
                    _, buffer = cv2.imencode('.jpg', frame_to_show)
                    await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("[INFO] 클라이언트 연결 종료됨.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Websocket Communication Error: {str(e)}")
    finally:
        if websocket in clients:
            clients.remove(websocket)

# 파일 경로에 대해 FPS 계산 작업을 처리하고, 작업 후 파일을 삭제하는 엔드포인트
@app.post("/fps-video/")
async def fps_video_endpoint(request: VideoPathRequest2):
    
    file_path=request.video_path
    file_path = urlparse(file_path)
    file_path = file_path.path.lstrip('/')  # 앞쪽 '/' 제거
    
    # 컨테이너 이름이 포함되어 있으면 제거
    if file_path.startswith(AZURE_CONTAINER_NAME + '/'):
        file_path = file_path[len(AZURE_CONTAINER_NAME) + 1:]
    
    # BlobClient 생성 (SAS URL로 접근)
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=file_path)

    # 임시 파일 경로 생성
    temp_file_path = os.path.join(tempfile.gettempdir(), os.path.basename(file_path))

    try:
        # Azure Blob Storage에서 파일 다운로드
        with open(temp_file_path, "wb") as download_file:
            download_blob = blob_client.download_blob()
            download_file.write(download_blob.readall())

        # OpenCV를 사용하여 FPS 계산
        cap = cv2.VideoCapture(temp_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()  # 리소스 해제

        # 작업 후 임시 파일 삭제
        os.remove(temp_file_path)

        return {"fps": fps}

    except Exception as e:
        # 오류 발생 시 처리
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

    finally:
        # 임시 파일 삭제(실패나 성공 여부와 관계없이)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/upload-json/")
async def upload_json(file: UploadFile = File(...)):
    try:
        # JSON 파일 읽기
        contents = await file.read()
        data = json.loads(contents)  # JSON 데이터 파싱

        print(f"Received data: {data}")  # 서버에서 받은 데이터 출력

        # 처리 결과 반환
        return data

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Main function
def main():
    video_path = "input_video.mp4"  # Input video path

    process_video(video_path)
    print(f"Processed video saved")

if __name__ == "__main__":
    main()
