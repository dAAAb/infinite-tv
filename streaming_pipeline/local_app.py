from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import sys

from streaming_pipeline.models import StartStreamRequest, UpdateConfigRequest
from streaming_pipeline.streaming_service import StreamingService


load_dotenv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

app = FastAPI(title="Infinite TV Local")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = StreamingService()


@app.on_event("startup")
def setup() -> None:
    service.setup()


@app.post("/start_stream")
def start_stream(request: StartStreamRequest):
    return service.start_streaming(request)


@app.post("/update_config")
def update_config(request: UpdateConfigRequest):
    return service.update_config(request)


@app.post("/stop_stream")
def stop_stream():
    return service.stop_streaming()


@app.get("/metrics")
def metrics():
    return service.get_metrics()


@app.websocket("/metrics/ws")
async def metrics_ws(websocket: WebSocket) -> None:
    await service.handle_metrics_websocket(websocket)


@app.websocket("/webrtc")
async def webrtc(websocket: WebSocket) -> None:
    await service.handle_webrtc(websocket)
