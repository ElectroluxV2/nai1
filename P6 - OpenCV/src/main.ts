import { ImageSegmenterResult } from "@mediapipe/tasks-vision";
import { pEvent as promisifyEvent } from 'p-event';
import { getImageSegmenter } from './get-image-segmenter.ts';
import { legendColors } from './constants.ts';

const imageSegmenter = await getImageSegmenter();

// Get DOM elements
const video = document.createElement("video") as HTMLVideoElement;
video.autoplay = true;

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const canvasCtx = canvas.getContext("2d", {
  willReadFrequently: true
})!;

// Toggle button
const toggleMainLoopButton = document.getElementById("toggleMainLoopButton") as HTMLButtonElement;
let mainLoopRunning: Boolean = false;
// If webcam supported, add event listener to button.
if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
  toggleMainLoopButton.addEventListener("click", toggleMainLoop);
} else {
  alert("getUserMedia() is not supported by your browser");
}

async function toggleMainLoop(_event: MouseEvent) {
  if (mainLoopRunning) {
    mainLoopRunning = false;
    toggleMainLoopButton.innerText = "Enable";
    return;
  }

  mainLoopRunning = true;
  toggleMainLoopButton.innerText = "Disable";

  // Activate the live webcam stream.
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: true,
  });

  // Start imageSegmentation once data loaded
  await promisifyEvent(video, "loadeddata");

  // Adjust canvas dimensions
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Get segmentation from the webcam
  let lastWebcamTime = -1;
  while (mainLoopRunning) {
    // Do not predict same frame multiple times
    if (video.currentTime === lastWebcamTime) {
      console.debug('Waiting for next webcam frame');
      await new Promise(r => requestAnimationFrame(r)); // Wait for the next frame
      continue;
    }

    lastWebcamTime = video.currentTime;
    canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    const startTimeMs = performance.now();

    // Segmenting the stream, single frame at the time
    const result = await new Promise<ImageSegmenterResult>(r => imageSegmenter.segmentForVideo(video, startTimeMs, r));

    const { data: canvasImageData } = canvasCtx.getImageData(0, 0, video.videoWidth, video.videoHeight);

    const mask: Float32Array = result.categoryMask!.getAsFloat32Array();
    for (let maskIndex = 0, pixelIndex = 0; maskIndex < mask.length; ++maskIndex, pixelIndex += 4) {
      const maskVal = Math.round(mask[maskIndex] * 255.0);
      const legendColor = legendColors[maskVal % legendColors.length];
      canvasImageData[pixelIndex] = (legendColor[0] + canvasImageData[pixelIndex]) / 2;
      canvasImageData[pixelIndex + 1] = (legendColor[1] + canvasImageData[pixelIndex + 1]) / 2;
      canvasImageData[pixelIndex + 2] = (legendColor[2] + canvasImageData[pixelIndex + 2]) / 2;
      canvasImageData[pixelIndex + 3] = (legendColor[3] + canvasImageData[pixelIndex + 3]) / 2;
    }
    const uint8Array = new Uint8ClampedArray(canvasImageData.buffer);
    const dataNew = new ImageData(uint8Array, video.videoWidth, video.videoHeight);

    canvasCtx.putImageData(dataNew, 0, 0);
  }
}
