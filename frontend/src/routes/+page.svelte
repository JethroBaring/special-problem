<script lang="ts">
  let ws: any;
  let transcription = "Waiting for analysis...";
  let sentiment = "N/A";
  let confidence = "N/A";
  let videoElement: HTMLVideoElement | null = null;
  let trackElement: HTMLTrackElement | null = null;
  let textTrack: TextTrack | null = null; // This is the actual text track we need!

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      videoElement!.srcObject = stream;

      const mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
      ws = new WebSocket("ws://127.0.0.1:8000/test/ws/audio");

      ws.onopen = () => console.log("Connected to WebSocket server");

      ws.onmessage = (event: any) => {
        const result = JSON.parse(event.data);
        transcription = result.transcription;
        sentiment = result.sentiment;
        confidence = `${(result.confidence * 100).toFixed(2)}%`;

        // Ensure textTrack is initialized and add captions dynamically
        if (trackElement) {
          textTrack = trackElement.track; // Get the actual TextTrack object
          if (textTrack) {
            textTrack.mode = "showing"; // Ensure captions are visible
            textTrack.addCue(new VTTCue(0, 60, transcription)); // Update captions
          }
        }
      };

      mediaRecorder.start(1000); // Send video/audio chunks every second

      mediaRecorder.ondataavailable = (event) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(event.data); // Send audio-video chunk
        }
      };
    } catch (error) {
      console.error("Error accessing camera/microphone:", error);
    }
  }
</script>

<main class="container">
  <h1>🎤🎥 Real-Time Interview Analysis</h1>
  <video bind:this={videoElement} autoplay playsinline>
    <track kind="captions" label="Live Captions" srclang="en" default bind:this={trackElement} />
  </video>
  <button on:click={startRecording}>Start Recording</button>

  <div class="results">
    <p><strong>Transcription:</strong> {transcription}</p>
    <p><strong>Sentiment:</strong> {sentiment}</p>
    <p><strong>Confidence:</strong> {confidence}</p>
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
