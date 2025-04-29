from fastapi import FastAPI, HTTPException, Depends, Request, status, UploadFile, File, Form, BackgroundTasks,APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://jlasi17.github.io", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Configuration
client = AsyncIOMotorClient(
    os.getenv("MONGODB_URL"),
    tls=True,
    tlsAllowInvalidCertificates=True  # Only for development!
)
db = client.memorylane
scores_collection = db.scores

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
    """Recursively convert MongoDB documents to JSON-serializable format"""
    if doc is None:
        return None
    if isinstance(doc, (str, int, float, bool)):
        return doc
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, datetime):
        return doc.isoformat()
    if isinstance(doc, list):
        return [convert_mongo_doc(item) for item in doc]
    if isinstance(doc, dict):
        return {k: convert_mongo_doc(v) for k, v in doc.items()}
    # Handle other cases (like custom objects with __dict__)
    if hasattr(doc, '__dict__'):
        return convert_mongo_doc(doc.__dict__)
    return str(doc)  # Fallback for other types


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
    completed: bool = False

    class Config:
        validate_by_name = True


# Medications Models
class MedicationCreate(BaseModel):
    patient_id: str
    name: str
    time: List[str]
    duration: int
    notes: str = None

# TODO:
class MedicationResponse(BaseModel):
    id: str
    patient_id: str
    patient_name: str
    doctor_id: str
    name: str
    time: List[str]
    duration: Optional[int] = None
    notes: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    taken_times: List[dict] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }

class AppointmentStatusUpdate(BaseModel):
    completed: bool

class MedicationStatusUpdate(BaseModel):
    taken: bool
    time: str

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
        <p>Please login at: https://jlasi17.github.io/memolane/#/</p>
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
            {"caretakers": current_user["username"]},
            {"patient_id": current_user["username"]}
        ]
    })
    
    if not patient:
        raise HTTPException(
            status_code=404,
            detail="No patient found for this user"
        )
    
    return {
        "patient": convert_mongo_doc({
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient["gender"],
            "patient_id": patient["patient_id"],
            "alzheimer_stage": patient.get("alzheimer_stage", "unknown"),
            "appointments": patient.get("appointments", []),
            "medications": patient.get("medications", [])
        }),
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

    collections = await db.list_collection_names()
    print("Existing collections:", collections)
    
    if "scores" not in collections:
        await db.create_collection("scores")
        print("Created scores collection")
    
    if "game_users" not in collections:
        await db.create_collection("game_users")
        print("Created game_users collection")

@app.on_event("startup")
async def create_indexes():
    await db.patients.create_index([("patient_id", 1)])
    await db.patients.create_index([("alzheimer_stage", 1)])
    await db.patients.create_index([("last_scan_date", -1)])
    await db.appointments.create_index([("doctor_id", 1)])
    await db.appointments.create_index([("date", 1), ("time", 1)])
    await db.medications.create_index([("expires_at", 1)])
    await db.scores.create_index([("game_name", 1)])
    await db.scores.create_index([("score", -1)])
    await db.scores.create_index([("date", -1)])


@app.get("/api/notifications")
async def get_notifications(current_user: dict = Depends(get_current_user)):
    try:
        notifications = []
        async for notification in db.notifications.find(
            {"user_id": current_user["username"]}
        ).sort("created_at", -1).limit(100):
            notifications.append(convert_mongo_doc(notification))
        
        return notifications
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/notifications")
async def create_notification(notification: Notification, current_user: dict = Depends(get_current_user)):
    try:
        notification_dict = notification.dict()
        notification_dict["user_id"] = current_user["username"]
        result = await db.notifications.insert_one(notification_dict)
        return {"id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user_patients")
async def get_user_patients(current_user: dict = Depends(get_current_user)):
    try:
        patients = []
        async for patient in db.patients.find({"caretakers": current_user["username"]}):
            patients.append(convert_mongo_doc(patient))
        
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
    patient_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all appointments for the current user"""
    try:
        # Build query based on user role
        query = {}
        
        if current_user["role"] == "doctor":
            query["doctor_id"] = current_user["username"]
            if patient_id:
                query["patient_id"] = patient_id
        else:
            # For patients/family, only show their own appointments
            patient = await db.patients.find_one({
                "$or": [
                    {"user_id": current_user["username"]},
                    {"caretakers": current_user["username"]},
                    {"patient_id": current_user["username"]}
                ]
            })
            
            if not patient:
                raise HTTPException(
                    status_code=404,
                    detail="Patient not found or access denied"
                )
            
            query["patient_id"] = patient["patient_id"]

        appointments = []
        async for appt in db.appointments.find(query).sort([("date", 1), ("time", 1)]):
            patient = await db.patients.find_one({"patient_id": appt["patient_id"]})
            doctor = await db.users.find_one({"username": appt["doctor_id"]})
            appt_date = appt["date"]
            if isinstance(appt_date, datetime):
                appt_date = appt_date.strftime("%Y-%m-%d")
            appointments.append({
                "id": str(appt["_id"]),
                "patient_id": appt["patient_id"],
                "patient_name": patient["name"] if patient else "Unknown",
                "doctor_id": appt["doctor_id"],
                "doctor_name": doctor.get("full_name", doctor["username"]) if doctor else "Unknown",
                "date": appt_date,
                "time": appt["time"],
                "description": appt.get("description", ""),
                "created_at": appt["created_at"],
                "completed": appt.get("completed", False)  # Include completed status
            })
        
        return appointments

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.put("/api/appointments/{appointment_id}/status")
async def update_appointment_status(
    appointment_id: str,
    status: AppointmentStatusUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Validate appointment_id
        try:
            appointment_oid = ObjectId(appointment_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid appointment ID format")

        # Verify appointment exists and user has access
        appointment = await db.appointments.find_one({
            "_id": appointment_oid,
            "$or": [
                {"patient_id": current_user["username"]},  # Patient can mark their own appointments
                {"doctor_id": current_user["username"]}    # Doctor can mark their appointments
            ]
        })
        
        if not appointment:
            raise HTTPException(status_code=404, detail="Appointment not found or access denied")

        # Update status
        await db.appointments.update_one(
            {"_id": appointment_oid},
            {"$set": {"completed": status.completed}}
        )

        return {"success": True, "message": "Appointment status updated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/medications/{medication_id}/status")
async def update_medication_status(
    medication_id: str,
    status: MedicationStatusUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Validate medication_id
        try:
            medication_oid = ObjectId(medication_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid medication ID format")

        # Verify medication exists and user has access
        medication = await db.medications.find_one({
            "_id": medication_oid,
            "patient_id": current_user["username"]  # Only patient can mark medications as taken
        })
        
        if not medication:
            raise HTTPException(status_code=404, detail="Medication not found or access denied")

        # Update status - track each time it's taken
        update_data = {
            "$push": {
                "taken_times": {
                    "time": status.time,
                    "taken_at": datetime.utcnow()
                }
            }
        }

        await db.medications.update_one(
            {"_id": medication_oid},
            update_data
        )

        return {"success": True, "message": "Medication status updated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Medications Endpoints
from datetime import datetime

@app.get("/api/medications", response_model=List[MedicationResponse])
async def get_medications(
    patient_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        query = {}
        
        if current_user["role"] == "doctor":
            query["doctor_id"] = current_user["username"]
            if patient_id:
                query["patient_id"] = patient_id
        else:
            patient = await db.patients.find_one({
                "$or": [
                    {"user_id": current_user["username"]},
                    {"caretakers": current_user["username"]},
                    {"patient_id": current_user["username"]}
                ]
            })
            
            if not patient:
                raise HTTPException(
                    status_code=404,
                    detail="Patient not found or access denied"
                )
            
            query["patient_id"] = patient["patient_id"]

        medications = []
        async for med in db.medications.find(query).sort("created_at", -1):
            patient = await db.patients.find_one({"patient_id": med["patient_id"]})
            
            time_list = med.get("time", [])
            if isinstance(time_list, str):
                time_list = [time_list]

            created_at = med.get("created_at")
            expires_at = med.get("expires_at")
            
            med_data = {
                "id": str(med["_id"]),
                "patient_id": med["patient_id"],
                "patient_name": patient["name"] if patient else "Unknown",
                "doctor_id": med["doctor_id"],
                "name": med["name"],
                "time": time_list,
                "duration": med.get("duration", 0),
                "notes": med.get("notes", ""),
                "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
                "expires_at": expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at,
                "taken_times": med.get("taken_times", [])  # Include taken times
            }
            medications.append(MedicationResponse(**med_data))
        
        return medications

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

        if  not medication.duration or medication.duration < 1:
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


# Add this to your models section
class GameUser(BaseModel):
    patient_id: str
    name: str
    level: int = 1
    exp: int = 0
    badges: List[str] = Field(default_factory=list)
    games_played: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = datetime.utcnow()
    last_played: Optional[datetime] = None

# Add these endpoints
@app.post("/api/game_user/initialize")
async def initialize_game_user(
    patient_id: str = Form(...),
    name: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Initialize a game user profile if it doesn't exist"""
    try:
        # Check if patient exists and user has access
        patient = await db.patients.find_one({
            "patient_id": patient_id,
            "$or": [
                {"user_id": current_user["username"]},
                {"caretakers": current_user["username"]}
            ]
        })
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found or access denied")

        # Check if game user already exists
        existing = await db.game_users.find_one({"patient_id": patient_id})
        if existing:
            return convert_mongo_doc(existing)

        # Create new game user
        game_user = {
            "patient_id": patient_id,
            "name": name,
            "level": 1,
            "exp": 0,
            "badges": [],
            "games_played": {},
            "created_at": datetime.utcnow()
        }

        result = await db.game_users.insert_one(game_user)
        game_user["_id"] = result.inserted_id
        
        return convert_mongo_doc(game_user)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/game_user/{patient_id}")
async def get_game_user(
    patient_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get game user profile"""
    try:
        print(f"Fetching game user for patient: {patient_id}")
        print(f"Current user: {current_user['username']}")

        # Verify patient exists and user has access - using same logic as patient_stats
        patient = await db.patients.find_one({
            "$or": [
                {"user_id": current_user["username"]},
                {"caretakers": current_user["username"]},
                {"patient_id": current_user["username"]}
            ],
            "patient_id": patient_id  # Also ensure we're getting the requested patient
        })
        
        if not patient:
            print(f"Patient {patient_id} not found or access denied for user {current_user['username']}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found or access denied"
            )

        # Check for existing game user
        game_user = await db.game_users.find_one({"patient_id": patient_id})
        
        if not game_user:
            print(f"Creating new game user for patient {patient_id}")
            game_user = {
                "patient_id": patient_id,
                "name": patient.get("name", "Player"),
                "level": 1,
                "exp": 0,
                "badges": [],
                "games_played": {},
                "created_at": datetime.utcnow()
            }
            
            try:
                result = await db.game_users.insert_one(game_user)
                game_user["_id"] = result.inserted_id
                print(f"Created new game user: {game_user}")
            except Exception as insert_error:
                print(f"Error creating game user: {str(insert_error)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create game profile"
                )

        return convert_mongo_doc(game_user)
        
    except HTTPException:
        # Re-raise HTTPExceptions (like our 404)
        raise
    except Exception as e:
        print(f"ERROR in get_game_user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

class Score(BaseModel):
    patient_id: str
    game_name: str
    score: int
    rounds_completed: int
    date: datetime = Field(default_factory=datetime.utcnow)
    is_high_score: bool = False

class PatientStatsResponse(BaseModel):
    patient: dict
    stats: dict
    
    class Config:
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }

@app.post("/api/save-score")
async def save_score(
    score_data: dict,
    current_user: dict = Depends(get_current_user)
):
    print(f"Attempting to save score for user: {current_user['username']}")
    print(f"Received score data: {score_data}")
    
    try:
        # Find patient record
        patient_record = await db.game_users.find_one({
            "patient_id": current_user["username"]
        })
        
        if not patient_record:
            # Initialize game user if not found
            patient_record = {
                "patient_id": current_user["username"],
                "name": "Player",
                "level": 1,
                "exp": 0,
                "badges": [],
                "games_played": {"memotap": 0, "high_score": 0},
                "created_at": datetime.utcnow()
            }
            await db.game_users.insert_one(patient_record)
        
        # Check if games_played exists and is an object
        if "games_played" not in patient_record or not isinstance(patient_record.get("games_played"), dict):
            await db.game_users.update_one(
                {"patient_id": current_user["username"]},
                {"$set": {"games_played": {"memotap": 0, "high_score": 0}}}
            )

        # Check for existing high score
        high_score = await db.scores.find_one(
            {"patient_id": current_user["username"], "game_name": "memotap"},
            sort=[("score", -1)]
        )

        is_high_score = not high_score or score_data["score"] > high_score["score"]
        # Calculate EXP to add (10 EXP per point scored)
        exp_to_add = score_data["score"] * 10
        current_exp = patient_record.get("exp", 0)
        current_level = patient_record.get("level", 1)
        
        # Calculate new EXP and handle level ups
        new_exp = current_exp + exp_to_add
        levels_gained = 0
        
        # Calculate how many levels should be gained
        while new_exp >= current_level * 100:
            new_exp -= current_level * 100
            current_level += 1
            levels_gained += 1

        # Save the score
        score_doc = {
            "patient_id": current_user["username"],
            "game_name": "memotap",
            "score": score_data["score"],
            "rounds_completed": score_data["rounds_completed"],
            "date": datetime.utcnow(),
            "is_high_score": is_high_score
        }
        result = await db.scores.insert_one(score_doc)

        # Prepare update data
        update_data = {
            "$inc": {
                "games_played.memotap": 1,
                "level": levels_gained
            },
            "$set": {
                "exp": new_exp,
                "last_played": datetime.utcnow()
            }
        }
        
        if is_high_score:
            update_data["$max"] = {"games_played.high_score": score_data["score"]}

        # Add badge if player leveled up
        if levels_gained > 0:
            badge_name = f"Level {current_level} Achiever"
            update_data["$addToSet"] = {"badges": badge_name}

        await db.game_users.update_one(
            {"patient_id": current_user["username"]},
            update_data
        )

        return {
            "message": "Score saved successfully",
            "current_level": current_level,
            "current_exp": new_exp,
            "exp_needed": current_level * 100,
            "leveled_up": levels_gained > 0,
            "new_level": current_level if levels_gained > 0 else None
        }

    except Exception as e:
        print(f"Error saving score: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/high-scores")
async def get_patient_high_scores(current_user: dict = Depends(get_current_user)):
    """Get high scores for the current user's patient"""
    try:
        # Get patient ID for the current user
        patient = await db.patients.find_one({
            "$or": [
                {"user_id": current_user["username"]},
                {"caretakers": current_user["username"]},
                {"patient_id": current_user["username"]}
            ]
        })
        
        if not patient:
            return JSONResponse(
                status_code=200,
                content={"scores": []}
            )
        
        # Get top 10 scores for this patient
        scores = await db.scores.find(
            {"patient_id": patient["patient_id"]},
            {"_id": 0, "game_name": 1, "score": 1, "date": 1, "is_high_score": 1}
        ).sort("score", -1).limit(10).to_list(None)
        
        return {"scores": scores or []}
        
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"scores": []}
        )

@app.get("/api/game_user/current")
async def get_current_game_user(current_user: dict = Depends(get_current_user)):
    """Get the current user's game profile"""
    try:
        # Use the current user's username as the patient_id
        patient_id = current_user["username"]
        
        game_user = await db.game_users.find_one({"patient_id": patient_id})
        if not game_user:
            # Create a default profile if none exists
            game_user = {
                "patient_id": patient_id,
                "name": current_user.get("name", "Player"),
                "level": 1,
                "exp": 0,
                "badges": [],
                "games_played": {},
                "created_at": datetime.utcnow(),
                "last_played": None
            }
            await db.game_users.insert_one(game_user)
        print("assadsas",game_user.level)
        return convert_mongo_doc(game_user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/save-memory-score")
async def save_memory_score(
    score_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Save score for memory matching game with level-based difficulty"""
    try:
        # Validate input
        required_fields = ["score", "level", "time", "matches", "difficulty"]
        if not all(key in score_data for key in required_fields):
            raise HTTPException(status_code=400, detail="Missing required score data")
        
        # Get or create game user profile
        game_user = await db.game_users.find_one({"patient_id": current_user["username"]})
        
        if not game_user:
            game_user = {
                "patient_id": current_user["username"],
                "name": current_user.get("name", "Player"),
                "level": 1,
                "exp": 0,
                "badges": [],
                "games_played": {"memory_match": 0},
                "created_at": datetime.utcnow()
            }
            await db.game_users.insert_one(game_user)

        # Check if this is a new high score for this difficulty
        high_score = await db.scores.find_one(
            {
                "patient_id": current_user["username"],
                "game_name": "memory_match",
                "difficulty": score_data["difficulty"]
            },
            sort=[("score", -1)]
        )

        is_high_score = not high_score or score_data["score"] > high_score["score"]
        
        # Calculate EXP based on performance (matches + time bonus)
        base_exp = score_data["matches"] * 5
        time_bonus = min(50, max(0, (score_data["time_limit"] - score_data["time"]) // 10))
        exp_to_add = base_exp + time_bonus
        
        current_exp = game_user.get("exp", 0)
        current_level = game_user.get("level", 1)
        
        # Handle level progression
        new_exp = current_exp + exp_to_add
        levels_gained = 0
        
        while new_exp >= current_level * 100:
            new_exp -= current_level * 100
            current_level += 1
            levels_gained += 1

        # Save the score
        score_doc = {
            "patient_id": current_user["username"],
            "game_name": "memory_match",
            "score": score_data["score"],
            "level": score_data["level"],
            "difficulty": score_data["difficulty"],
            "time": score_data["time"],
            "time_limit": score_data["time_limit"],
            "matches": score_data["matches"],
            "moves": score_data.get("moves", 0),
            "date": datetime.utcnow(),
            "is_high_score": is_high_score
        }
        await db.scores.insert_one(score_doc)

        # Update player profile
        update_data = {
            "$inc": {
                f"games_played.memory_match": 1,
                "exp": exp_to_add
            },
            "$set": {
                "level": current_level,
                "last_played": datetime.utcnow()
            }
        }
        
        if is_high_score:
            update_data["$set"][f"high_scores.memory_match_{score_data['difficulty']}"] = score_data["score"]
        
        if levels_gained > 0:
            update_data["$addToSet"] = {
                "badges": {
                    "$each": [f"Memory Master L{current_level - i}" for i in range(levels_gained)]
                }
            }

        await db.game_users.update_one(
            {"patient_id": current_user["username"]},
            update_data
        )

        return {
            "success": True,
            "new_level": current_level,
            "exp_gained": exp_to_add,
            "is_high_score": is_high_score,
            "levels_gained": levels_gained
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory-high-scores")
async def get_memory_high_scores(
    difficulty: Optional[str] = None,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get high scores for memory matching game, optionally filtered by difficulty"""
    try:
        query = {
            "patient_id": current_user["username"],
            "game_name": "memory_match"
        }
        
        if difficulty:
            query["difficulty"] = difficulty
        
        scores = await db.scores.find(
            query,
            {
                "_id": 0,
                "score": 1,
                "level": 1,
                "difficulty": 1,
                "time": 1,
                "time_limit": 1,
                "matches": 1,
                "moves": 1,
                "date": 1
            }
        ).sort([("score", -1), ("time", 1)]).limit(limit).to_list(None)
        
        return {"scores": scores or []}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))