import asyncio
import base64
import cv2
import numpy as np
from fastapi import WebSocket
import traceback
import sys
import json

# Import only the DeepFace-based facial analysis function
from app.models.facial import analyze_facial_emotion
from app.models.posture import analyze_posture

async def process_realtime_video(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected for real-time analysis")
    sys.stdout.flush()

    frame_skip = 5
    frame_count = 0
    results = []

    try:
        while True:
            print("⏳ Waiting for client data...")
            sys.stdout.flush()
            
            # Receive data from client
            try:
                data = await websocket.receive_text()
                print(f"📥 Received data length: {len(data) if data else 0}")
                sys.stdout.flush()
            except Exception as e:
                print(f"❌ Error receiving data: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                break
            
            # Skip corrupted or empty frames
            if not data or len(data) < 100:
                print("⚠️ Skipping invalid video data")
                sys.stdout.flush()
                continue

            frame_count += 1
            print(f"🔢 Processing frame #{frame_count}")
            sys.stdout.flush()

            # Skip frames to reduce load
            if frame_count % frame_skip != 0:
                print(f"⏭️ Skipping frame {frame_count} (not divisible by {frame_skip})")
                sys.stdout.flush()
                continue

            # Process as base64 data
            try:
                print("🔍 Starting facial analysis...")
                sys.stdout.flush()
                
                # Use the DeepFace-based analysis
                face_data = await analyze_facial_emotion(data)
                
                print(f"😀 Facial analysis result: {face_data}")
                sys.stdout.flush()
                
                print("🔍 Starting posture analysis...")
                sys.stdout.flush()
                # Posture analysis (async)
                posture_data = await analyze_posture(data)
                print(f"🧍 Posture analysis result: {posture_data}")
                sys.stdout.flush()
                
                # Combine results
                result = {
                    "frame": frame_count,
                    "emotion": face_data.get("dominant_emotion", "N/A"),
                    "emotion_score": face_data.get("emotion_score", 0),
                    "posture": posture_data.get("posture_feedback", "N/A"),
                    "posture_score": posture_data.get("confidence", 0)
                }
                
                # Add to results queue
                results.append(result)
                
                # Keep queue size manageable
                if len(results) > 5:
                    results.pop(0)
                
                print(f"✅ Analysis result: {result}")
                sys.stdout.flush()
                
                # Send back the analysis
                response_data = {
                    "frames": results,
                    "dominant_emotion": face_data.get("dominant_emotion", "N/A")
                }
                
                # Serialize and send JSON
                try:
                    json_str = json.dumps(response_data)
                    print(f"📤 Sending response: {json_str[:100]}...")
                    sys.stdout.flush()
                    await websocket.send_text(json_str)
                    print("📤 Response sent successfully")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"❌ Error sending response: {e}")
                    traceback.print_exc()
                    sys.stdout.flush()
                
            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                try:
                    await websocket.send_json({"error": str(e)})
                except:
                    print("❌ Could not send error message to client")
                    sys.stdout.flush()

    except Exception as e:
        print(f"❌ Error in WebSocket: {e}")
        traceback.print_exc()
        sys.stdout.flush()

    finally:
        print("⚠️ WebSocket closed")
        sys.stdout.flush()
        try:
            await websocket.close()
        except:
            print("❌ Error closing websocket")
            sys.stdout.flush()