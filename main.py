from fastapi import FastAPI, HTTPException, Depends, Request, status, UploadFile, File, Form, BackgroundTasks, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
import os
import bcrypt
from datetime import datetime, timedelta
import jwt
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr, field_validator
import re
import io
import random
import string
import uuid
from datetime import datetime, timedelta
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Load environment variables
load_dotenv()

# Email configuration (add to your .env file)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Initialize FastAPI
app = FastAPI()


cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  secure=True
)

UPLOAD_DIR = "uploads/images"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Configure upload directory
UPLOAD_DIR2 = "uploads/mri_scans"
Path(UPLOAD_DIR2).mkdir(parents=True, exist_ok=True)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# MongoDB Configuration
client = AsyncIOMotorClient(
    os.getenv("MONGODB_URL"),
    tls=True,
    tlsAllowInvalidCertificates=True  # Only for development!
)
db = client.memorylane

# Security Configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

stage_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
def load_model(model_path="../models/resnet18_alzheimer_model.pth"):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(stage_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

def convert_mongo_doc(doc):
    if doc is None:
        return None
    if isinstance(doc, list):
        return [convert_mongo_doc(item) for item in doc]
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, dict):
        return {k: convert_mongo_doc(v) for k, v in doc.items()}
    return doc

# Pydantic Models
class User(BaseModel):
    username: str
    email: str
    role: str  # 'family', 'doctor', or 'patient'
    password: str

class Patient(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, example="John Doe")
    age: int = Field(..., gt=0, lt=120, example=65)
    gender: str = Field(..., pattern="^(male|female|other)$", example="male")
    medical_history: Optional[str] = Field(None, max_length=1000, example="Hypertension, Diabetes")
    email: EmailStr = Field(..., example="patient@example.com")
    phone: str = Field(..., min_length=10, max_length=15, example="8801177005")
    user_id: Optional[str] = None
    patient_id: Optional[str] = None  # New 6-digit patient ID
    passcode: Optional[str] = None
    caretakers: List[str] = []
    stage: Optional[str] = Field(None, pattern="^(non_demented|mild|moderate|severe|unknown)$")
    appointments: List[str] = Field(default_factory=list)  # Added for appointments
    medications: List[str] = Field(default_factory=list)   # Added for medications

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        if not re.match(r'^[0-9]{10,15}$', v):
            raise ValueError('Phone must be 10-15 digits')
        return v

class LoginForm(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class Notification(BaseModel):
    user_id: str
    message: str
    read: bool = False
    created_at: datetime = datetime.utcnow()

class UserResponse(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = True

class ImageResponse(BaseModel):
    id: str
    patient_id: str
    description: str
    upload_date: datetime
    content_type: str
    image_url: Optional[str] = None

class MRIScanBase(BaseModel):
    patient_id: str
    scan_date: datetime
    file_path: str
    original_filename: str
    file_size: int
    notes: Optional[str] = None
    uploaded_by: str
    uploaded_at: datetime

class MRIScanCreate(MRIScanBase):
    pass

class MRIScanInDB(MRIScanBase):
    id: str

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Appointments Models
class AppointmentCreate(BaseModel):
    patient_id: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    description: str = None

class AppointmentResponse(BaseModel):
    id: str
    patient_id: str
    patient_name: str
    doctor_id: str
    date: str
    time: str
    description: str
    created_at: datetime

    class Config:
        validate_by_name = True


# Medications Models
class MedicationCreate(BaseModel):
    patient_id: str
    name: str
    time: List[str]
    duration: int
    notes: str = None

class MedicationResponse(BaseModel):
    id: str
    patient_id: str
    patient_name: str
    doctor_id: str
    name: str
    time: List[str]
    duration: int
    notes: str
    created_at: datetime
    expires_at: datetime


class ScoreData(BaseModel):
    player_name: str
    score: int
    rounds_completed: int


# Helper Functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return await db.users.find_one({"username": username})
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise credentials_exception

def generate_patient_id():
    return ''.join(random.choices(string.digits, k=6))

def generate_passcode():
    return ''.join(random.choices(string.digits, k=8))

async def predict_alzheimer(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            prediction = stage_names[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "status": "completed"
        }
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

# Routes
@app.get("/")
async def root():
    return {
        "message": "Memory Lane API", 
        "status": "running",
        "docs": "http://127.0.0.1:8000/docs"
    }

@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "message": "Endpoint not found",
            "available_endpoints": [
                "/register",
                "/login",
                "/patients",
                "/api/register_patient",
                "/api/patient_stats"
            ]
        }
    )

@app.post("/register")
async def register(user: User):
    if await db.users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
    
    result = await db.users.insert_one({
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "hashed_password": hashed.decode()
    })
    
    return {"id": str(result.inserted_id)}

@app.post("/login")
async def login(form: LoginForm):
    user = await db.users.find_one({"username": form.username})
    
    if not user or not bcrypt.checkpw(form.password.encode(), user["hashed_password"].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(days=7)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "role": user["role"]
        }
    }

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await db.users.find_one({"username": form_data.username})
    if not user or not bcrypt.checkpw(form_data.password.encode(), user["hashed_password"].encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

async def send_patient_credentials(email: str, patient_id: str, passcode: str):
    try:
        print(f"Attempting to send email to: {email}")
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = "Memory Lane - Your Patient Credentials"
        
        body = f"""
        <h2>Welcome to Memory Lane</h2>
        <p><strong>Patient ID:</strong> {patient_id}</p>
        <p><strong>Temporary Passcode:</strong> {passcode}</p>
        <p>Please login at: http://localhost:3000/login</p>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, email, msg.as_string())
            print("Email successfully sent!")
            
        return True
    except Exception as e:
        print(f"SMTP Error Details: {str(e)}")
        if hasattr(e, 'smtp_error'):
            print(f"SMTP Server Error: {e.smtp_error.decode()}")
        return False
                   
@app.post("/api/register_patient")
async def register_patient(patient: Patient, current_user: dict = Depends(get_current_user)):
    try:
        if patient.patient_id:
            existing_patient = await db.patients.find_one({"patient_id": patient.patient_id})
            if not existing_patient:
                raise HTTPException(status_code=404, detail="Patient ID not found")
            
            await db.patients.update_one(
                {"patient_id": patient.patient_id},
                {"$addToSet": {"caretakers": current_user["username"]}}
            )
            
            return {"success": True, "message": "Patient linked successfully"}
        if await db.users.find_one({"email": patient.email}):
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        patient_id = generate_patient_id()
        passcode = generate_passcode()
        hashed_passcode = bcrypt.hashpw(passcode.encode(), bcrypt.gensalt()).decode()
        
        patient_user = {
            "username": patient_id,
            "email": patient.email,
            "role": "patient",
            "hashed_password": hashed_passcode,
            "created_at": datetime.utcnow()
        }
        
        patient_data = patient.model_dump()
        patient_data.update({
            "patient_id": patient_id,
            "passcode": passcode,
            "caretakers": [current_user["username"]],
            "created_at": datetime.utcnow(),
            "appointments": [],
            "medications": []
        })
        del patient_data["user_id"]

        async with await client.start_session() as session:
            async with session.start_transaction():
                await db.users.insert_one(patient_user, session=session)
                await db.patients.insert_one(patient_data, session=session)
        
        email_sent = await send_patient_credentials(
            patient.email,
            patient_id,
            passcode
        )
        
        if not email_sent:
            print(f"Failed to send email to {patient.email}")
        
        return {
            "success": True,
            "patient_id": patient_id,
            "passcode": passcode,
            "message": "Patient registered successfully. Credentials sent to email."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
@app.get("/api/patient_stats")
async def get_patient_stats(current_user: dict = Depends(get_current_user)):
    patient = await db.patients.find_one({
        "$or": [
            {"user_id": current_user["username"]},
            {"caretakers": current_user["username"]}
        ]
    })
    
    if not patient:
        raise HTTPException(
            status_code=404,
            detail="No patient found for this user"
        )
    
    return {
        "patient": {
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient["gender"],
            "stage": patient.get("stage", "unknown")
        },
        "stats": {
            "last_updated": datetime.utcnow(),
            "medication_adherence": patient.get("adherence", 0)
        }
    }

@app.get("/patients", response_model=List[Patient])
async def get_patients(current_user: dict = Depends(get_current_user)):
    if current_user["role"] == "doctor":
        patients = await db.patients.find().to_list(100)
    else:
        patients = await db.patients.find({
            "$or": [
                {"user_id": current_user["username"]},
                {"caretakers": current_user["username"]}
            ]
        }).to_list(100)
    
    return patients

# Startup Event
@app.on_event("startup")
async def startup_db_client():
    await db.command("ping")
    print("Successfully connected to MongoDB!")

@app.on_event("startup")
async def create_indexes():
    await db.patients.create_index([("patient_id", 1)])
    await db.patients.create_index([("alzheimer_stage", 1)])
    await db.patients.create_index([("last_scan_date", -1)])
    await db.appointments.create_index([("doctor_id", 1)])
    await db.appointments.create_index([("patient_id", 1)])
    await db.appointments.create_index([("date", 1), ("time", 1)])
    await db.medications.create_index([("patient_id", 1)])
    await db.medications.create_index([("expires_at", 1)])

@app.post("/api/notifications")
async def create_notification(notification: Notification, current_user: dict = Depends(get_current_user)):
    notification_dict = notification.dict()
    notification_dict["user_id"] = current_user["username"]
    result = await db.notifications.insert_one(notification_dict)
    return {"id": str(result.inserted_id)}

@app.get("/api/notifications")
async def get_notifications(current_user: dict = Depends(get_current_user)):
    notifications = await db.notifications.find(
        {"user_id": current_user["username"]}
    ).sort("created_at", -1).to_list(100)
    return notifications

@app.get("/api/user_patients")
async def get_user_patients(current_user: dict = Depends(get_current_user)):
    try:
        patients = []
        async for patient in db.patients.find({"caretakers": current_user["username"]}):
            patient["_id"] = str(patient["_id"])
            patients.append(patient)
        
        return {"patients": patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/user")
async def get_user(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}
    
@app.post("/api/upload_image")
async def upload_image(
    image: UploadFile = File(...),
    description: str = Form(...),
    patient_id: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        if not image or image.filename == '':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image file selected"
            )

        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only image files are allowed"
            )

        contents = await image.read()
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            contents,
            folder=f"memorylane/{patient_id}",
            public_id=f"img_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        image_doc = {
            "patient_id": patient_id,
            "description": description,
            "original_filename": image.filename,
            "content_type": image.content_type,
            "file_size": len(contents),
            "uploaded_by": current_user["username"],
            "uploaded_at": datetime.utcnow(),
            "cloudinary_url": upload_result["secure_url"],
            "cloudinary_public_id": upload_result["public_id"]
        }

        result = await db.images.insert_one(image_doc)

        return {
            "success": True,
            "message": "Image uploaded successfully",
            "image_id": str(result.inserted_id),
            "patient_id": patient_id,
            "filename": image.filename,
            "url": upload_result["secure_url"],
            "file_size": len(contents),
            "uploaded_at": datetime.utcnow().isoformat()
        }

    except HTTPException as he:
        return {
            "success": False,
            "message": he.detail,
            "status_code": he.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Upload failed: {str(e)}",
            "status_code": 500
        }


@app.get("/api/images/{patient_id}")
async def get_patient_images(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    patient = await db.patients.find_one({
        "patient_id": patient_id,
        "$or": [
            {"user_id": current_user["username"]},
            {"caretakers": current_user["username"]}
        ]
    })
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found or access denied"
        )

    images = []
    async for img in db.images.find({"patient_id": patient_id}).sort("uploaded_at", -1):
        images.append({
            "id": str(img["_id"]),
            "patient_id": img["patient_id"],
            "description": img["description"],
            "url": img["cloudinary_url"],
            "original_filename": img["original_filename"],
            "uploaded_at": img["uploaded_at"],
            "file_size": img["file_size"]
        })
    
    return {"images": images}



@app.delete("/api/images/{image_id}")
async def delete_image(
    image_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        img = await db.images.find_one({"_id": ObjectId(image_id)})
        
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")

        patient = await db.patients.find_one({
            "patient_id": img["patient_id"],
            "$or": [
                {"user_id": current_user["username"]},
                {"caretakers": current_user["username"]}
            ]
        })
        
        if not patient:
            raise HTTPException(status_code=403, detail="Access denied")

        # Delete from Cloudinary
        if "cloudinary_public_id" in img:
            cloudinary.uploader.destroy(img["cloudinary_public_id"])

        await db.images.delete_one({"_id": ObjectId(image_id)})

        return {"success": True, "message": "Image deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload_mri_scan")
async def upload_mri_scan(
    background_tasks: BackgroundTasks,
    scan_file: UploadFile = File(...),
    patient_id: str = Form(...),
    scan_date: str = Form(...),
    notes: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        patient = await db.patients.find_one({
            "patient_id": patient_id,
            "$or": [
                {"user_id": current_user["username"]},
                {"caretakers": current_user["username"]}
            ]
        })
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found or access denied")

        valid_extensions = ('.jpg', '.jpeg', '.png', '.dicom', '.nii', '.nii.gz')
        if not scan_file.filename.lower().endswith(valid_extensions):
            raise HTTPException(status_code=400, 
                              detail=f"Only {', '.join(valid_extensions)} files allowed")

        contents = await scan_file.read()
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            contents,
            folder=f"memorylane/mri_scans/{patient_id}",
            public_id=f"mri_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            resource_type="raw"  # Important for non-image files
        )

        scan_doc = {
            "patient_id": patient_id,
            "scan_date": datetime.strptime(scan_date, "%Y-%m-%d"),
            "original_filename": scan_file.filename,
            "file_size": len(contents),
            "notes": notes,
            "uploaded_by": current_user["username"],
            "uploaded_at": datetime.utcnow(),
            "processing_status": "pending",
            "cloudinary_url": upload_result["secure_url"],
            "cloudinary_public_id": upload_result["public_id"],
            "alzheimer_prediction": {
                "status": "pending"
            }
        }

        result = await db.mri_scans.insert_one(scan_doc)
        scan_id = str(result.inserted_id)
        
        background_tasks.add_task(
            process_mri_prediction,
            contents,
            scan_id,
            patient_id,
            current_user["username"]
        )
        
        return {
            "success": True,
            "message": "MRI scan uploaded. Processing started.",
            "scan_id": scan_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_mri_prediction(image_bytes: bytes, scan_id: str, patient_id: str, username: str):
    try:
        await db.mri_scans.update_one(
            {"_id": ObjectId(scan_id)},
            {"$set": {
                "processing_status": "processing",
                "alzheimer_prediction.status": "processing"
            }}
        )
        
        prediction_result = await predict_alzheimer(image_bytes)
        
        stage = None
        if prediction_result["prediction"] == "NonDemented":
            stage = "0"
        elif prediction_result["prediction"] == "VeryMildDemented":
            stage = "1"
        elif prediction_result["prediction"] == "MildDemented":
            stage = "2"
        elif prediction_result["prediction"] == "ModerateDemented":
            stage = "3"
        
        update_data = {
            "processing_status": "completed",
            "alzheimer_prediction": {
                "status": "completed",
                "prediction": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "stage": stage,
                "completed_at": datetime.utcnow()
            }
        }
        
        await db.mri_scans.update_one(
            {"_id": ObjectId(scan_id)},
            {"$set": update_data}
        )
        
        await db.patients.update_one(
            {"patient_id": patient_id},
            {
                "$set": {
                    "alzheimer_stage": stage,
                    "last_scan_date": datetime.utcnow(),
                    "last_scan_id": scan_id
                },
                "$push": {
                    "scan_history": {
                        "scan_id": scan_id,
                        "date": datetime.utcnow(),
                        "stage": stage,
                        "prediction": prediction_result["prediction"]
                    }
                }
            }
        )
        
    except Exception as e:
        await db.mri_scans.update_one(
            {"_id": ObjectId(scan_id)},
            {"$set": {
                "processing_status": f"failed: {str(e)}",
                "alzheimer_prediction.status": f"failed: {str(e)}"
            }}
        )
        raise

@app.get("/api/mri_scan/{scan_id}")
async def get_mri_scan(scan_id: str, current_user: dict = Depends(get_current_user)):
    try:
        scan = await db.mri_scans.find_one(
            {"_id": ObjectId(scan_id)},
            projection={
                "patient_id": 1,
                "scan_date": 1,
                "file_path": 1,
                "processing_status": 1,
                "alzheimer_prediction": 1,
                "uploaded_at": 1
            }
        )
        
        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")
            
        patient = await db.patients.find_one({
            "patient_id": scan["patient_id"],
            "$or": [
                {"user_id": current_user["username"]},
                {"caretakers": current_user["username"]}
            ]
        })
        
        if not patient:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return convert_mongo_doc(scan)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/patients/{patient_id}")
async def get_patient(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        patient = await db.patients.find_one({
            "patient_id": patient_id,
            "$or": [
                {"user_id": current_user["username"]},
                {"caretakers": current_user["username"]}
            ]
        }, projection={
            "_id": 0,
            "patient_id": 1,
            "name": 1,
            "alzheimer_stage": 1,
            "last_scan_date": 1,
            "scan_history": 1
        })
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found or access denied")
            
        return convert_mongo_doc(patient)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Appointments Endpoints
@app.get("/api/appointments", response_model=List[AppointmentResponse])
async def get_appointments(
    current_user: dict = Depends(get_current_user)
):
    """Get all appointments for the current doctor"""
    appointments = []
    async for appt in db.appointments.find(
        {"doctor_id": current_user["username"]}
    ).sort([("date", 1), ("time", 1)]):
        patient = await db.patients.find_one({"patient_id": appt["patient_id"]})
        appointments.append({
            "id": str(appt["_id"]),  # Transform _id to id
            "patient_id": appt["patient_id"],
            "patient_name": patient["name"] if patient else "Unknown",
            "doctor_id": appt["doctor_id"],
            "date": appt["date"],
            "time": appt["time"],
            "description": appt.get("description", ""),
            "created_at": appt["created_at"]
        })
    return appointments


@app.post("/api/appointments", response_model=AppointmentResponse)
async def create_appointment(
    appointment: AppointmentCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new appointment"""
    try:
        # Validate date and time formats
        appointment_date = datetime.strptime(appointment.date, "%Y-%m-%d").date()
        datetime.strptime(appointment.time, "%H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date or time format")
    
    # Check if patient exists
    patient = await db.patients.find_one({"patient_id": appointment.patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get patient name safely
    patient_name = patient.get("name", "Unknown Patient")  # Fallback if name doesn't exist
    
    # Check for time slot conflicts
    existing_appt = await db.appointments.find_one({
        "doctor_id": current_user["username"],
        "date": appointment.date,
        "time": appointment.time
    })
    
    if existing_appt:
        raise HTTPException(status_code=400, detail="Time slot already booked")
    
    # Create new appointment
    new_appt = {
        "patient_id": appointment.patient_id,
        "doctor_id": current_user["username"],
        "date": appointment.date,
        "time": appointment.time,
        "description": appointment.description,
        "created_at": datetime.utcnow()
    }
    
    result = await db.appointments.insert_one(new_appt)
    created_appt = await db.appointments.find_one({"_id": result.inserted_id})
    
    # Return response with safe patient name
    return {
        "id": str(created_appt["_id"]),
        "patient_id": created_appt["patient_id"],
        "patient_name": patient_name,  # Use the safely obtained name
        "doctor_id": created_appt["doctor_id"],
        "date": created_appt["date"],
        "time": created_appt["time"],
        "description": created_appt.get("description", ""),
        "created_at": created_appt["created_at"]
    }


@app.delete("/api/appointments/{appointment_id}")
async def delete_appointment(
    appointment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Complete/delete an appointment"""
    try:
        # Validate the appointment_id
        obj_id = ObjectId(appointment_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid appointment ID format")
    
    # Find and verify the appointment
    appointment = await db.appointments.find_one({
        "_id": obj_id,
        "doctor_id": current_user["username"]
    })
    
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    try:
        # Remove from patient's appointments
        await db.patients.update_one(
            {"patient_id": appointment["patient_id"]},
            {"$pull": {"appointments": appointment_id}}
        )
        
        # Delete the appointment
        await db.appointments.delete_one({"_id": obj_id})
        
        return {"message": "Appointment completed/deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Medications Endpoints
@app.get("/api/medications", response_model=List[MedicationResponse])
async def get_medications(
    current_user: dict = Depends(get_current_user)
):
    medications = []
    async for med in db.medications.find(
        {"doctor_id": current_user["username"]}
    ).sort("created_at", -1):
        patient = await db.patients.find_one({"patient_id": med["patient_id"]})
        expires_at = med["created_at"] + timedelta(days=med["duration"])
        med_dict = {
            **med,
            "id": str(med["_id"]),
            "patient_name": patient["name"] if patient else "Unknown",
            "expires_at": expires_at
        }
        del med_dict["_id"]
        medications.append(med_dict)
    return medications

@app.post("/api/medications", response_model=MedicationResponse)
async def create_medication(
    medication: MedicationCreate,
    current_user: dict = Depends(get_current_user)
):
    try:
        valid_times = {"Morning", "Afternoon", "Evening"}
        if not medication.time or any(t not in valid_times for t in medication.time):
            raise HTTPException(
                status_code=400,
                detail=f"Time must be any of: {', '.join(valid_times)}"
            )

        if medication.duration < 1:
            raise HTTPException(
                status_code=400,
                detail="Duration must be at least 1 day"
            )

        patient = await db.patients.find_one({
            "patient_id": medication.patient_id,
            "caretakers": current_user["username"]
        })

        if not patient:
            raise HTTPException(
                status_code=404,
                detail="Patient not found or you don't have permission to prescribe for this patient"
            )

        now = datetime.utcnow()
        new_med = {
            "patient_id": medication.patient_id,
            "doctor_id": current_user["username"],
            "name": medication.name,
            "time": medication.time,
            "duration": medication.duration,
            "notes": medication.notes,
            "created_at": now,
            "expires_at": now + timedelta(days=medication.duration)
        }

        result = await db.medications.insert_one(new_med)
        med_id = str(result.inserted_id)

        await db.patients.update_one(
            {"patient_id": medication.patient_id},
            {"$addToSet": {"medications": med_id}}
        )

        return {
            "id": med_id,
            "patient_id": medication.patient_id,
            "patient_name": patient.get("name", "Unknown"),
            "doctor_id": current_user["username"],
            "name": medication.name,
            "time": medication.time,
            "duration": medication.duration,
            "notes": medication.notes,
            "created_at": now,
            "expires_at": now + timedelta(days=medication.duration)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create medication: {str(e)}"
        )


@app.delete("/api/medications/{medication_id}")
async def delete_medication(
    medication_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Remove a medication prescription"""
    try:
        # Validate medication_id
        if not medication_id:
            raise HTTPException(status_code=400, detail="Medication ID is required")

        try:
            medication_oid = ObjectId(medication_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid medication ID format")

        # Verify medication exists
        medication = await db.medications.find_one({
            "_id": medication_oid,
            "doctor_id": current_user["username"]
        })
        
        if not medication:
            raise HTTPException(status_code=404, detail="Medication not found")

        # Perform deletion
        await db.medications.delete_one({"_id": medication_oid})
        
        # Remove from patient's medications list
        await db.patients.update_one(
            {"patient_id": medication["patient_id"]},
            {"$pull": {"medications": medication_id}}
        )
        
        return {"success": True, "message": "Medication deleted successfully","deleted_id": medication_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background task to clean expired medications
async def clean_expired_medications():
    """Should be called periodically (e.g., daily)"""
    try:
        now = datetime.utcnow()
        expired_meds = db.medications.find({
            "expires_at": {"$lt": now}
        })
        
        async for med in expired_meds:
            # Remove from patient's medications list
            await db.patients.update_one(
                {"patient_id": med["patient_id"]},
                {"$pull": {"medications": str(med["_id"])}}
            )
            
            # Delete the medication
            await db.medications.delete_one({"_id": med["_id"]})
            
        print(f"Cleaned up expired medications at {now}")
        
    except Exception as e:
        print(f"Error cleaning expired medications: {str(e)}")



@app.post("/save-score")
async def save_score(score_data: ScoreData):
    # Save current score
    scores_collection.insert_one(score_data.dict())
    
    # Check if it's a high score
    high_score = scores_collection.find_one(sort=[("score", -1)])
    
    return {
        "message": "Score saved successfully",
        "is_high_score": high_score["_id"] == score_data.score
    }

@app.get("/high-scores")
async def get_high_scores(limit: int = 10):
    scores = list(scores_collection.find(
        {},
        {"_id": 0, "player_name": 1, "score": 1, "rounds_completed": 1}
    ).sort("score", -1).limit(limit))
    return {"scores": scores}

@app.get("/generate-sequence")
async def generate_sequence(round_number: int):
    # Determine number of colors based on round
    if round_number <= 5:
        num_colors = 4
    elif round_number <= 10:
        num_colors = 6
    elif round_number <= 15:
        num_colors = 8
    elif round_number <= 20:
        num_colors = 10
    elif round_number <= 23:
        num_colors = 16
    else:
        num_colors = 25
    
    # Generate sequence of taps (length = round number)
    sequence = [random.randint(0, num_colors - 1) for _ in range(round_number)]
    
    return {
        "sequence": sequence,
        "num_colors": num_colors
    }

