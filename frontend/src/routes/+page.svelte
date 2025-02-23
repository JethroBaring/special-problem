<script lang="ts">
  let wsVideo: WebSocket;
  let videoElement: HTMLVideoElement | null = null;
  let canvas: HTMLCanvasElement | null = null;
  let emotion = "N/A";

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true }); // Removed audio
      videoElement!.srcObject = stream;

      wsVideo = new WebSocket("ws://127.0.0.1:8000/test/ws/video");

      wsVideo.onopen = () => console.log("✅ Connected to Video WebSocket");
      wsVideo.onerror = (error) => console.error("❌ WebSocket error:", error);
      wsVideo.onclose = () => console.log("⚠️ Video WebSocket closed.");
      wsVideo.onmessage = (event: MessageEvent) => processVideoMessage(event);

      canvas = document.createElement("canvas");
      const videoTrack = stream.getVideoTracks()[0];

      // Capture and send a frame every second
      setInterval(() => captureFrame(), 1000);
    } catch (error) {
      console.error("❌ Error accessing camera:", error);
    }
  }

  function captureFrame() {
    if (!canvas || !videoElement || !wsVideo || wsVideo.readyState !== WebSocket.OPEN) return;

    const ctx = canvas.getContext("2d");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    ctx?.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Convert to base64 image and send
    canvas.toBlob((blob) => {
      if (blob) {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64data = reader.result?.toString();
          if (base64data) wsVideo.send(base64data);
        };
        reader.readAsDataURL(blob);
      }
    }, "image/jpeg");
  }

  function stopRecording() {
    wsVideo?.close();
    videoElement?.srcObject?.getTracks().forEach((track) => track.stop());
  }

  function processVideoMessage(event: MessageEvent) {
    try {
      const result = JSON.parse(event.data);
      emotion = result.dominant_emotion || "N/A";
      console.log("📥 Received video analysis:", result);
    } catch (error) {
      console.error("❌ Error processing video analysis:", error);
    }
  }
</script>

<main class="container">
  <h1>🎥 Real-Time Emotion Analysis</h1>
  <video bind:this={videoElement} autoplay playsinline></video>
  <button on:click={startRecording}>Start Analysis</button>
  <button on:click={stopRecording}>Stop</button>

  <div class="results">
    <p><strong>Emotion:</strong> {emotion}</p>
  </div>
</main>

<style>
  .container {
    text-align: center;
    font-family: sans-serif;
  }
  video {
    width: 100%;
    max-width: 400px;
    border-radius: 10px;
    margin-top: 10px;
  }
  .results {
    margin-top: 20px;
    border: 1px solid #ddd;
    padding: 10px;
    display: inline-block;
  }
</style>
