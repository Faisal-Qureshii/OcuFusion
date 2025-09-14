import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from train import start_teacher_training, start_student_training, predict_image, TrainingStatus, evaluate_model

app = FastAPI(title='MultiEYE Backend', version='1.1')

status = TrainingStatus()
_last_eval = {}

class TrainRequest(BaseModel):
    phase: str  # 'teacher' or 'student'
    data_dir: str
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 32
    extra: Optional[dict] = None

@app.post('/train/start')
async def start_train(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.phase not in ('teacher','student'):
        raise HTTPException(status_code=400, detail='phase must be teacher or student')
    status.stop = False
    if req.phase == 'teacher':
        background_tasks.add_task(start_teacher_training, req.data_dir, req.epochs, req.batch_size, status, req.extra or {})
    else:
        background_tasks.add_task(start_student_training, req.data_dir, req.epochs, req.batch_size, status, req.extra or {})
    return JSONResponse({'message':'training started', 'phase':req.phase})

@app.get('/train/status')
async def train_status():
    return status.get_status()

@app.post('/train/stop')
async def stop_train():
    status.request_stop()
    return JSONResponse({'message':'stop requested'})

@app.post('/predict')
async def predict(file: UploadFile = File(...), modality: str = 'fundus'):
    data = await file.read()
    pred = predict_image(data, modality)
    return JSONResponse({'prediction': pred})

@app.get('/download/checkpoint')
async def download_checkpoint(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='file not found')
    return FileResponse(path, filename=os.path.basename(path))

@app.get('/analysis')
async def get_analysis():
    return status.get_results()

class EvalRequest(BaseModel):
    model_path: str
    data_dir: str
    subset: Optional[str] = 'dev'
    modality: Optional[str] = 'fundus'
    img_size: Optional[int] = 224
    batch_size: Optional[int] = 32

@app.post('/eval/run')
async def run_evaluation(req: EvalRequest, background_tasks: BackgroundTasks):
    def _run():
        global _last_eval
        metrics = evaluate_model(req.model_path, req.data_dir, subset=req.subset, modality=req.modality, img_size=req.img_size, batch_size=req.batch_size, analysis_dir='./analysis')
        _last_eval = metrics
        status.last_eval = metrics
    background_tasks.add_task(_run)
    return JSONResponse({'message':'evaluation started, check /eval/results shortly'})

@app.get('/eval/results')
async def eval_results():
    if status.last_eval:
        return status.last_eval
    else:
        raise HTTPException(status_code=404, detail='no evaluation results found')
