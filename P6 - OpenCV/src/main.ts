import { ImageSegmenter, ImageSegmenterResult, FilesetResolver } from "@mediapipe/tasks-vision";
import { pEvent as promisifyEvent } from 'p-event';

const wasm = import.meta.glob('/node_modules/@mediapipe/tasks-vision/wasm/*', {
  as: 'url',
  eager: true,
});

const firstWasmModulePath = wasm[Object.keys(wasm)[0]];
const wasmModulesBasePath = firstWasmModulePath.substring(0, firstWasmModulePath.lastIndexOf('/'));
const wasmFileset = await FilesetResolver.forVisionTasks(wasmModulesBasePath);

// Get DOM elements
const video = document.getElementById("webcam") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const canvasCtx = canvas.getContext("2d", {
  willReadFrequently: true
})!;

const imageSegmenter = await ImageSegmenter.createFromOptions(wasmFileset, {
  baseOptions: {
    modelAssetPath: "/selfie_multiclass_256x256.tflite",
    delegate: "GPU",
  },
  runningMode: "VIDEO",
  outputCategoryMask: true,
  outputConfidenceMasks: false,
});

const labels = imageSegmenter.getLabels();

// RGBA
const legendColors = [
  [255, 197, 0, 255], // Vivid Yellow
  [128, 62, 117, 255], // Strong Purple
  [255, 104, 0, 255], // Vivid Orange
  [166, 189, 215, 255], // Very Light Blue
  [193, 0, 32, 255], // Vivid Red
  [206, 162, 98, 255], // Grayish Yellow
  [129, 112, 102, 255], // Medium Gray
  [0, 125, 52, 255], // Vivid Green
  [246, 118, 142, 255], // Strong Purplish Pink
  [0, 83, 138, 255], // Strong Blue
  [255, 112, 92, 255], // Strong Yellowish Pink
  [83, 55, 112, 255], // Strong Violet
  [255, 142, 0, 255], // Vivid Orange Yellow
  [179, 40, 81, 255], // Strong Purplish Red
  [244, 200, 0, 255], // Vivid Greenish Yellow
  [127, 24, 13, 255], // Strong Reddish Brown
  [147, 170, 0, 255], // Vivid Yellowish Green
  [89, 51, 21, 255], // Deep Yellowish Brown
  [241, 58, 19, 255], // Vivid Reddish Orange
  [35, 44, 22, 255], // Dark Olive Green
  [0, 161, 194, 255] // Vivid Blue
];

// Webcam button
const toggleMainLoopButton = document.getElementById("webcamButton") as HTMLButtonElement;
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
