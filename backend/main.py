from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from indicators import run_analysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 나중에 배포 시 프론트엔드 URL로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    ticker: str        # 예: "IBM"
    holding_days: int  # 예: 5
    years: int         # 예: 2

@app.get("/")
def root():
    return {"status": "ok", "message": "Stock Indicator API is running"}

@app.post("/api/analyze")
def analyze(req: AnalysisRequest):
    try:
        result = run_analysis(req.ticker, req.holding_days, req.years)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
