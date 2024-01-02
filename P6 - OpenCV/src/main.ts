import { ImageSegmenter, ImageSegmenterResult, FilesetResolver } from "@mediapipe/tasks-vision";

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
const enableWebcamButton = document.getElementById("webcamButton") as HTMLButtonElement;
let webcamRunning: Boolean = false;
// If webcam supported, add event listener to button.
if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  alert("getUserMedia() is not supported by your browser");
}

async function enableCam(_event: MouseEvent) {
  if (webcamRunning) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE SEGMENTATION";
    video.removeEventListener("loadeddata", predictWebcam);
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE SEGMENTATION";
    // Start imageSegmentation once data loaded
    video.addEventListener("loadeddata", predictWebcam);
  }

  // Activate the live webcam stream.
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: true,
  });
}

// Get segmentation from the webcam
let lastWebcamTime = -1;
async function predictWebcam() {
  // Do not predict if webcam is not running
  if (!webcamRunning) return;

  // Do not predict same frame multiple times
  if (video.currentTime === lastWebcamTime) {
    return requestAnimationFrame(predictWebcam); // Predict next frame
  }

  lastWebcamTime = video.currentTime;
  canvasCtx!.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  const startTimeMs = performance.now();

  // Start segmenting the stream.
  imageSegmenter.segmentForVideo(video, startTimeMs, callbackForVideo);
}

function callbackForVideo(result: ImageSegmenterResult) {
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

  window.requestAnimationFrame(predictWebcam);
}
