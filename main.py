from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
from tangoflux import TangoFluxInference
import os
from pydub import AudioSegment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
OUTPUT_FILE = "output.wav"

def get_model():
    global model
    if model is None:
        model = TangoFluxInference(name="declare-lab/TangoFlux")
    return model

@app.get("/generate")
async def generate(prompt: str):
    model_instance = get_model()
    audio = model_instance.generate(prompt, steps=50, duration=10)
    torchaudio.save(OUTPUT_FILE, audio, 44100)
    return JSONResponse({"status": "success"})

@app.get("/preview")
async def preview():
    return FileResponse(OUTPUT_FILE, media_type="audio/wav")

@app.get("/save")
async def save():
    return FileResponse(
        OUTPUT_FILE,
        media_type="audio/wav",
        filename=os.path.basename(OUTPUT_FILE),
        headers={"Content-Disposition": f"attachment; filename={os.path.basename(OUTPUT_FILE)}"}
    )

@app.get("/trim-preview")
async def trim(start: float, end: float):
    audio = AudioSegment.from_wav(OUTPUT_FILE)
    trimmed = audio[start * 1000.0 : end * 1000.0]
    trimmed.export("trimmed.wav", format="wav")
    return FileResponse("trimmed.wav", media_type="audio/wav")

@app.get("/trim-save")
async def save():
    return FileResponse(
        "trimmed.wav",
        media_type="audio/wav",
        filename=os.path.basename("trimmed.wav"),
        headers={"Content-Disposition": f"attachment; filename={os.path.basename('trimmed.wav')}"}
    )