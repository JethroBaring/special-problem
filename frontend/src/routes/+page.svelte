<script lang="ts">
  let wsVideo: WebSocket;
  let videoElement: HTMLVideoElement | null = null;
  let canvas: HTMLCanvasElement | null = null;
  let emotion = "N/A";
  let isConnected = false;
  let connectionStatus = "Not connected";
  let debugLog: string[] = [];
  
  function addDebugLog(message: string) {
    const timestamp = new Date().toLocaleTimeString();
    debugLog = [...debugLog, `[${timestamp}] ${message}`];
    console.log(message);
    
    // Keep log to a reasonable size
    if (debugLog.length > 20) {
      debugLog = debugLog.slice(debugLog.length - 20);
    }
  }

  async function startRecording() {
    try {
      addDebugLog("Starting camera access...");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      addDebugLog("Camera access granted");
      
      if (videoElement) {
        videoElement.srcObject = stream;
        addDebugLog("Video stream attached to video element");
      }

      connectionStatus = "Connecting...";
      addDebugLog("Opening WebSocket connection...");
      wsVideo = new WebSocket("ws://127.0.0.1:8000/test/ws/video");

      wsVideo.onopen = () => {
        addDebugLog("✅ Connected to Video WebSocket");
        isConnected = true;
        connectionStatus = "Connected";
        startFrameCapture();
      };
      
      wsVideo.onerror = (error) => {
        addDebugLog(`❌ WebSocket error: ${JSON.stringify(error)}`);
        connectionStatus = "Connection error";
      };
      
      wsVideo.onclose = () => {
        addDebugLog("⚠️ Video WebSocket closed");
        isConnected = false;
        connectionStatus = "Disconnected";
      };
      
      wsVideo.onmessage = (event: MessageEvent) => {
        addDebugLog(`📥 Received message: ${event.data.substring(0, 50)}...`);
        processVideoMessage(event);
      };

      canvas = document.createElement("canvas");
      addDebugLog("Canvas created for frame capture");
      
    } catch (error) {
      addDebugLog(`❌ Error accessing camera: ${error}`);
      connectionStatus = "Camera error";
    }
  }

  function startFrameCapture() {
    addDebugLog("Starting frame capture...");
    // Capture and send a frame every second
    const intervalId = setInterval(() => {
      if (!isConnected) {
        addDebugLog("WebSocket disconnected, stopping frame capture");
        clearInterval(intervalId);
        return;
      }
      captureFrame();
    }, 1000);
    addDebugLog("Frame capture interval set (1 second)");
  }

  function captureFrame() {
    if (!canvas) {
      addDebugLog("❌ Canvas not available");
      return;
    }
    
    if (!videoElement) {
      addDebugLog("❌ Video element not available");
      return;
    }
    
    if (!wsVideo || wsVideo.readyState !== WebSocket.OPEN) {
      addDebugLog(`❌ WebSocket not ready (state: ${wsVideo?.readyState})`);
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      addDebugLog("❌ Could not get canvas context");
      return;
    }
    
    // Check if video is playing and has dimensions
    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      addDebugLog("❌ Video dimensions not available yet");
      return;
    }
    
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    addDebugLog(`📏 Canvas size set to ${canvas.width}x${canvas.height}`);
    
    try {
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      addDebugLog("✅ Frame captured to canvas");
      
      canvas.toBlob((blob) => {
        if (blob) {
          addDebugLog(`📦 Blob created: ${blob.size} bytes`);
          
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64data = reader.result?.toString().split(",")[1];
            if (base64data) {
              addDebugLog(`📸 Base64 length: ${base64data.length}`);
              wsVideo.send(base64data);
              addDebugLog("📤 Frame sent via WebSocket");
            } else {
              addDebugLog("❌ Failed to extract base64 data");
            }
          };
          reader.readAsDataURL(blob);
        } else {
          addDebugLog("❌ Failed to create blob from canvas");
        }
      }, "image/jpeg", 0.8);
    } catch (error) {
      addDebugLog(`❌ Error capturing frame: ${error}`);
    }
  }

  function stopRecording() {
    addDebugLog("Stopping recording...");
    
    if (wsVideo) {
      wsVideo.close();
      addDebugLog("WebSocket closed");
      isConnected = false;
      connectionStatus = "Disconnected";
    }
    
    if (videoElement && videoElement.srcObject) {
      const tracks = videoElement.srcObject as MediaStream;
      tracks.getTracks().forEach((track) => {
        track.stop();
        addDebugLog(`Stopped track: ${track.kind}`);
      });
    }
    
    addDebugLog("Recording stopped");
  }

  function processVideoMessage(event: MessageEvent) {
    try {
      const result = JSON.parse(event.data);
      addDebugLog(`Parsed message: ${JSON.stringify(result).substring(0, 100)}...`);
      
      if (result.error) {
        addDebugLog(`❌ Server error: ${result.error}`);
        return;
      }
      
      if (result.frames && result.frames.length > 0) {
        // Get the latest frame data
        const latestFrame = result.frames[result.frames.length - 1];
        emotion = latestFrame.emotion || "N/A";
        addDebugLog(`Updated emotion to: ${emotion}`);
      } else if (result.dominant_emotion) {
        // Handle direct emotion response format
        emotion = result.dominant_emotion;
        addDebugLog(`Updated emotion to: ${emotion}`);
      }
    } catch (error) {
      addDebugLog(`❌ Error processing video analysis: ${error}`);
    }
  }
  
  function clearLogs() {
    debugLog = [];
  }
</script>

<main class="container">
  <h1>🎥 Real-Time Emotion Analysis</h1>
  <video bind:this={videoElement} autoplay playsinline>
    <track kind="captions" srclang="en" label="English captions" />
  </video>
  <div class="controls">
    <button on:click={startRecording} disabled={isConnected}>Start Analysis</button>
    <button on:click={stopRecording} disabled={!isConnected}>Stop</button>
  </div>

  <div class="results">
    <p><strong>Emotion:</strong> {emotion}</p>
    <p><strong>Status:</strong> {connectionStatus}</p>
  </div>
  
  <div class="debug-panel">
    <h3>Debug Log</h3>
    <button on:click={clearLogs}>Clear Logs</button>
    <div class="log-container">
      {#each debugLog as logEntry}
        <div class="log-entry">{logEntry}</div>
      {/each}
    </div>
  </div>
</main>

<style>
  .container {
    text-align: center;
    font-family: sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
  }
  video {
    width: 100%;
    max-width: 400px;
    border-radius: 10px;
    margin-top: 10px;
  }
  .controls {
    margin-top: 10px;
  }
  .results {
    margin-top: 20px;
    border: 1px solid #ddd;
    padding: 10px;
    display: inline-block;
  }
  .debug-panel {
    margin-top: 30px;
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 10px;
    text-align: left;
  }
  .log-container {
    max-height: 300px;
    overflow-y: auto;
    background-color: #f5f5f5;
    padding: 10px;
    border-radius: 3px;
    font-family: monospace;
    font-size: 12px;
  }
  .log-entry {
    margin-bottom: 4px;
    border-bottom: 1px solid #eee;
    padding-bottom: 4px;
  }
</style>