#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import threading
import uuid
from typing import Dict, Optional

import numpy as np
import torch
import torchaudio
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import wenet
from silero_vad import load_silero_vad, VADIterator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'silero-vad-wenet-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

SAMPLING_RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLING_RATE * CHUNK_DURATION_MS / 1000)

vad_model = None
wenet_model = None
sessions: Dict[str, dict] = {}
sessions_lock = threading.Lock()


def init_models(model_name: str = 'firered', device: str = 'cpu'):
    global vad_model, wenet_model
    print("Loading Silero VAD model...")
    vad_model = load_silero_vad()
    print(f"Loading WeNet model: {model_name} on {device}...")
    wenet_model = wenet.load_model(model_name, device=device)
    print("Models loaded successfully!")


class SpeechSession:
    def __init__(self, session_id: str, vad_params: dict):
        self.session_id = session_id
        self.vad_params = vad_params
        self.vad_iterator = VADIterator(
            vad_model,
            threshold=vad_params.get('threshold', 0.5),
            sampling_rate=SAMPLING_RATE,
            min_silence_duration_ms=vad_params.get('min_silence_duration_ms', 100),
            speech_pad_ms=vad_params.get('speech_pad_ms', 30),
        )
        self.audio_buffer = []
        self.is_speaking = False
        self.speech_segments = []
        self.current_segment = []
        self.device = next(wenet_model.parameters()).device
        
    def process_audio_chunk(self, audio_data: np.ndarray) -> Optional[dict]:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        
        self.audio_buffer.append(audio_data.copy())
        
        speech_dict = self.vad_iterator(audio_data, return_seconds=True)
        
        result = None
        
        if speech_dict is not None:
            if 'start' in speech_dict:
                self.is_speaking = True
                self.current_segment = [audio_data.copy()]
                socketio.emit('speech_start', {
                    'session_id': self.session_id,
                    'timestamp': speech_dict['start']
                }, room=self.session_id)
            elif 'end' in speech_dict:
                self.is_speaking = False
                if self.current_segment:
                    self.current_segment.append(audio_data.copy())
                    full_audio = np.concatenate(self.current_segment)
                    self.speech_segments.append(full_audio)
                    
                    recognition_result = self._recognize_segment(full_audio)
                    
                    socketio.emit('speech_end', {
                        'session_id': self.session_id,
                        'timestamp': speech_dict['end'],
                        'text': recognition_result.get('text', '')
                    }, room=self.session_id)
                    
                    result = recognition_result
                    self.current_segment = []
        elif self.is_speaking:
            self.current_segment.append(audio_data.copy())
        
        return result
    
    def _recognize_segment(self, audio_data: np.ndarray) -> dict:
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save(temp_path, audio_tensor, SAMPLING_RATE)
            
            result = wenet_model.transcribe(temp_path)
            
            os.unlink(temp_path)
            
            return {
                'text': result.text,
                'tokens': result.tokens if hasattr(result, 'tokens') else [],
                'confidence': result.confidence if hasattr(result, 'confidence') else 1.0
            }
        except Exception as e:
            print(f"Recognition error: {e}")
            return {'text': '', 'error': str(e)}
    
    def reset(self):
        self.vad_iterator.reset_states()
        self.audio_buffer = []
        self.is_speaking = False
        self.speech_segments = []
        self.current_segment = []
    
    def update_vad_params(self, vad_params: dict):
        self.vad_params = vad_params
        self.vad_iterator = VADIterator(
            vad_model,
            threshold=vad_params.get('threshold', 0.5),
            sampling_rate=SAMPLING_RATE,
            min_silence_duration_ms=vad_params.get('min_silence_duration_ms', 100),
            speech_pad_ms=vad_params.get('speech_pad_ms', 30),
        )


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'connected'})


@socketio.on('start_session')
def handle_start_session(data: dict):
    session_id = str(uuid.uuid4())
    vad_params = data.get('vad_params', {})
    
    with sessions_lock:
        sessions[session_id] = SpeechSession(session_id, vad_params)
    
    emit('session_started', {
        'session_id': session_id,
        'vad_params': vad_params
    })
    print(f"Session started: {session_id}")


@socketio.on('audio_data')
def handle_audio_data(data: dict):
    session_id = data.get('session_id')
    audio_bytes = data.get('audio_data')
    
    if not session_id or not audio_bytes:
        emit('error', {'message': 'Invalid audio data'})
        return
    
    with sessions_lock:
        session = sessions.get(session_id)
    
    if not session:
        emit('error', {'message': 'Session not found'})
        return
    
    try:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        num_chunks = len(audio_data) // CHUNK_SIZE
        for i in range(num_chunks):
            chunk = audio_data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
            result = session.process_audio_chunk(chunk)
            
            if result and result.get('text'):
                emit('recognition_result', {
                    'session_id': session_id,
                    'text': result['text'],
                    'is_final': True
                })
        
        remaining = len(audio_data) % CHUNK_SIZE
        if remaining > 0:
            remaining_chunk = audio_data[-remaining:]
            padded_chunk = np.pad(remaining_chunk, (0, CHUNK_SIZE - remaining), 'constant')
            result = session.process_audio_chunk(padded_chunk)
            
            if result and result.get('text'):
                emit('recognition_result', {
                    'session_id': session_id,
                    'text': result['text'],
                    'is_final': True
                })
                
    except Exception as e:
        print(f"Audio processing error: {e}")
        emit('error', {'message': str(e)})


@socketio.on('update_vad_params')
def handle_update_vad_params(data: dict):
    session_id = data.get('session_id')
    vad_params = data.get('vad_params', {})
    
    with sessions_lock:
        session = sessions.get(session_id)
    
    if not session:
        emit('error', {'message': 'Session not found'})
        return
    
    session.update_vad_params(vad_params)
    emit('vad_params_updated', {
        'session_id': session_id,
        'vad_params': vad_params
    })
    print(f"VAD params updated for session {session_id}: {vad_params}")


@socketio.on('stop_session')
def handle_stop_session(data: dict):
    session_id = data.get('session_id')
    
    with sessions_lock:
        session = sessions.pop(session_id, None)
    
    if session:
        if session.current_segment:
            full_audio = np.concatenate(session.current_segment)
            result = session._recognize_segment(full_audio)
            if result.get('text'):
                emit('recognition_result', {
                    'session_id': session_id,
                    'text': result['text'],
                    'is_final': True
                })
        
        emit('session_stopped', {'session_id': session_id})
        print(f"Session stopped: {session_id}")
    else:
        emit('error', {'message': 'Session not found'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


def get_args():
    parser = argparse.ArgumentParser(description='Silero VAD + WeNet Speech Recognition Server')
    parser.add_argument('--port', default=5000, type=int, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', type=str, help='Server host')
    parser.add_argument('--model', default='firered', type=str, help='WeNet model name')
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', 'cuda', 'npu'], help='Device for inference')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    init_models(model_name=args.model, device=args.device)
    print(f"Starting server on {args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)
