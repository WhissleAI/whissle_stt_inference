#!/usr/bin/env python3
import asyncio
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import uvicorn

from .engine import ASREngine
from .streaming import StreamingSession, StreamingConfig
from .vad import load_silero_vad, SileroVAD
from .config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "en-meta"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024
MAX_WS_FRAME_SIZE = 1 * 1024 * 1024
MAX_CONCURRENT_WS_SESSIONS = 50

engines: Dict[str, ASREngine] = {}
engine: Optional[ASREngine] = None
silero_vad: Optional[SileroVAD] = None
startup_time: Optional[float] = None
_active_ws_sessions = 0

_cpu_count = multiprocessing.cpu_count()
_pool = ThreadPoolExecutor(max_workers=max(_cpu_count, settings.max_concurrent))


def _discover_models() -> dict:
    models = {}

    if os.path.isdir(settings.model_dir):
        model_onnx = os.path.join(settings.model_dir, 'model.onnx')
        if os.path.isfile(model_onnx):
            models[DEFAULT_MODEL_ID] = {
                'model_dir': settings.model_dir,
                'device': settings.device,
                'silence_padding': settings.silence_padding,
                'default_language': settings.default_language,
                'beam_width': settings.beam_width,
                'lm_alpha': settings.lm_alpha,
                'lm_beta': settings.lm_beta,
            }

    if settings.models_json:
        try:
            for m in json.loads(settings.models_json):
                mid = m['id']
                models[mid] = {
                    'model_dir': m['dir'],
                    'device': m.get('device', settings.device),
                    'silence_padding': float(m.get('silence_padding', settings.silence_padding)),
                    'default_language': m.get('default_language', settings.default_language),
                    'beam_width': int(m.get('beam_width', settings.beam_width)),
                    'lm_alpha': float(m.get('lm_alpha', settings.lm_alpha)),
                    'lm_beta': float(m.get('lm_beta', settings.lm_beta)),
                }
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Invalid ASR_MODELS JSON: %s", e)

    for key, val in os.environ.items():
        if key.startswith('ASR_MODEL_') and key.endswith('_DIR') and key != 'ASR_MODEL_DIR':
            model_onnx = os.path.join(val, 'model.onnx')
            if not os.path.isfile(model_onnx):
                continue
            mid = key[len('ASR_MODEL_'):-len('_DIR')].lower().replace('_', '-')
            if mid not in models:
                models[mid] = {
                    'model_dir': val,
                    'device': settings.device,
                    'silence_padding': settings.silence_padding,
                    'default_language': settings.default_language,
                    'beam_width': settings.beam_width,
                    'lm_alpha': settings.lm_alpha,
                    'lm_beta': settings.lm_beta,
                }

    return models


@asynccontextmanager
async def lifespan(app):
    global engine, engines, silero_vad, startup_time

    model_configs = _discover_models()
    print(f"Device: {settings.device}")
    print(f"Inference thread pool: {_pool._max_workers} workers")
    print(f"Models to load: {list(model_configs.keys())}")

    start = time.time()

    for model_id, mcfg in model_configs.items():
        print(f"\n--- Loading model '{model_id}' from: {mcfg['model_dir']} ---")
        eng = ASREngine(
            model_dir=mcfg['model_dir'],
            device=mcfg['device'],
            silence_padding=mcfg['silence_padding'],
            default_language=mcfg['default_language'],
            beam_width=mcfg['beam_width'],
            lm_alpha=mcfg['lm_alpha'],
            lm_beta=mcfg['lm_beta'],
        )
        engines[model_id] = eng
        print(f"Model '{model_id}' loaded — decoder: {eng.decoder_type}, vocab: {len(eng.vocabulary)}")

    startup_time = time.time() - start
    engine = next(iter(engines.values())) if engines else None
    print(f"\nAll models loaded in {startup_time:.2f}s")

    # Warmup
    import numpy as _np_warmup
    for model_id, eng in engines.items():
        try:
            silent = _np_warmup.zeros(16000 * 2, dtype=_np_warmup.float32)
            t0 = time.time()
            eng.transcribe(silent, sample_rate=16000, use_lm=False)
            print(f"Warmup '{model_id}': OK ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"Warmup '{model_id}': skipped ({e})")

    # Load Silero VAD from default model dir
    if engine:
        silero_vad = load_silero_vad(engine.model_dir)
        if silero_vad:
            print("Silero VAD: loaded")

    yield


app = FastAPI(
    title="Whissle ASR Inference Server",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)


def _get_engine(model_id: str = "") -> ASREngine:
    if not model_id:
        if engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return engine
    if model_id not in engines:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_id}'. Available: {list(engines.keys())}",
        )
    return engines[model_id]


@app.get("/")
async def demo_page():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return JSONResponse({"message": "Whissle ASR Inference Server", "docs": "/docs"})


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "whissle-asr-inference",
        "models_loaded": len(engines),
        "startup_time": f"{startup_time:.2f}s" if startup_time else None,
    }


@app.get("/models")
async def list_models():
    if not engines:
        raise HTTPException(status_code=503, detail="No models loaded")
    models_info = {}
    for mid, eng in engines.items():
        models_info[mid] = {
            "model_dir": str(eng.model_dir),
            "decoder_type": eng.decoder_type,
            "vocab_size": len(eng.vocabulary),
            "sample_rate": eng.sample_rate,
            "languages": eng.available_languages,
            "has_tag_classifier": eng._tag_classifier is not None,
        }
    return {"models": models_info}


@app.post("/asr/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    metadata_prob: bool = Form(False),
    top_k: int = Form(5),
    language: str = Form(""),
    use_lm: bool = Form(True),
    beam_width: Optional[int] = Form(None),
    lm_alpha: Optional[float] = Form(None),
    lm_beta: Optional[float] = Form(None),
    hotwords: str = Form(""),
    hotword_weight: float = Form(10.0),
    model: str = Form(""),
):
    eng = _get_engine(model)
    hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else None

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Audio file too large (max 50MB)")

        loop = asyncio.get_running_loop()
        start = time.time()
        result = await loop.run_in_executor(
            _pool,
            partial(
                eng.transcribe,
                audio_bytes,
                metadata_prob=metadata_prob,
                top_k=top_k,
                language=language or None,
                use_lm=use_lm,
                beam_width=beam_width,
                lm_alpha=lm_alpha,
                lm_beta=lm_beta,
                hotwords=hw_list,
                hotword_weight=hotword_weight,
            ),
        )
        result['inference_time'] = f"{time.time() - start:.3f}s"
        result['model'] = model or DEFAULT_MODEL_ID
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/asr/transcribe-url")
async def transcribe_url(
    url: str = Form(...),
    metadata_prob: bool = Form(False),
    language: str = Form(""),
    use_lm: bool = Form(True),
    model: str = Form(""),
):
    eng = _get_engine(model)

    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            audio_bytes = resp.read()
        if len(audio_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Audio too large")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    loop = asyncio.get_running_loop()
    start = time.time()
    result = await loop.run_in_executor(
        _pool,
        partial(
            eng.transcribe,
            audio_bytes,
            metadata_prob=metadata_prob,
            language=language or None,
            use_lm=use_lm,
        ),
    )
    result['inference_time'] = f"{time.time() - start:.3f}s"
    result['model'] = model or DEFAULT_MODEL_ID
    return result


@app.websocket("/asr/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Streaming ASR over WebSocket.

    Protocol:
      1. Client sends JSON config: {"type": "config", "language": "en", ...}
      2. Client sends binary PCM frames (Int16LE, 16kHz, mono)
      3. Server sends JSON transcript segments as they are ready
      4. Client sends {"type": "channel", "name": "system"} to tag audio
      5. Client sends {"type": "end"} to flush and close
    """
    global _active_ws_sessions

    if not engines:
        await websocket.close(code=1011, reason="Model not loaded")
        return

    if _active_ws_sessions >= MAX_CONCURRENT_WS_SESSIONS:
        await websocket.close(code=1013, reason="Too many concurrent sessions")
        return

    await websocket.accept()

    ws_engine: ASREngine = engine  # type: ignore
    session: Optional[StreamingSession] = None
    loop = asyncio.get_running_loop()
    flushed_cleanly = False
    conn_start = time.monotonic()

    _PCM_QUEUE_MAX = 500
    _PCM_HIGH_WATER = 400
    pcm_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=_PCM_QUEUE_MAX)
    cmd_queue: asyncio.Queue[dict] = asyncio.Queue()

    _active_ws_sessions += 1
    logger.info("WebSocket /asr/stream opened")

    async def _receiver():
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                    except (ValueError, TypeError):
                        await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                        continue
                    msg_type = data.get("type", "")
                    if msg_type == "keepalive":
                        continue
                    if msg_type in ("config", "channel"):
                        await cmd_queue.put(data)
                        continue
                    if msg_type == "end":
                        await cmd_queue.put(data)
                        break

                if "bytes" in message and message["bytes"]:
                    raw = message["bytes"]
                    if len(raw) > MAX_WS_FRAME_SIZE:
                        await websocket.send_json({"type": "error", "message": "Frame too large"})
                        continue
                    if pcm_queue.qsize() >= _PCM_HIGH_WATER:
                        dropped = 0
                        while pcm_queue.qsize() > _PCM_HIGH_WATER // 2:
                            try:
                                pcm_queue.get_nowait()
                                dropped += 1
                            except asyncio.QueueEmpty:
                                break
                        if dropped:
                            logger.warning("Backpressure: dropped %d frames", dropped)
                    await pcm_queue.put(raw)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.exception("WebSocket receiver error: %s", exc)
        finally:
            await pcm_queue.put(None)

    async def _processor():
        nonlocal ws_engine, session, flushed_cleanly

        while True:
            while not cmd_queue.empty():
                try:
                    cmd = cmd_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                msg_type = cmd.get("type", "")
                if msg_type == "config":
                    cfg = StreamingConfig.from_dict(cmd)
                    req_model = cmd.get("model", "")
                    if req_model and req_model in engines:
                        ws_engine = engines[req_model]
                    session = StreamingSession(ws_engine, cfg, silero_vad=silero_vad)
                    logger.info("Stream config: model=%s lang=%s lm=%s",
                                req_model or DEFAULT_MODEL_ID, cfg.language, cfg.use_lm)
                elif msg_type == "channel":
                    if session:
                        session.set_channel(cmd.get("name", "microphone"))
                elif msg_type == "end":
                    if session:
                        segments = await loop.run_in_executor(_pool, session.flush, True)
                        for seg in segments:
                            await websocket.send_json(seg.to_dict())
                    flushed_cleanly = True
                    await websocket.send_json({"type": "end"})
                    return

            first = await pcm_queue.get()
            if first is None:
                return

            frames = [first]
            while not pcm_queue.empty():
                try:
                    item = pcm_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    break
                frames.append(item)

            combined = b"".join(frames)
            if not combined:
                continue

            if session is None:
                session = StreamingSession(ws_engine, StreamingConfig(), silero_vad=silero_vad)

            segments = await loop.run_in_executor(_pool, session.feed, combined)
            for seg in segments:
                await websocket.send_json(seg.to_dict())

            if not segments and session.should_get_interim():
                interim = await loop.run_in_executor(_pool, session.get_interim)
                if interim is not None:
                    await websocket.send_json(interim.to_dict())

    try:
        await asyncio.gather(_receiver(), _processor())
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
    finally:
        _active_ws_sessions -= 1
        if not flushed_cleanly and session:
            try:
                segments = session.flush(force=True)
                for seg in segments:
                    await websocket.send_json(seg.to_dict())
            except Exception:
                pass
        elapsed = time.monotonic() - conn_start
        logger.info("WebSocket /asr/stream closed (%.1fs)", elapsed)


def main():
    uvicorn.run(
        "src.server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
    )


if __name__ == "__main__":
    main()
