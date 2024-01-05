/**
 * Author: Mateusz Budzisz
 * Based on https://codepen.io/mediapipe-preview/pen/xxJNjbN
 */

import { ImageSegmenterResult } from "@mediapipe/tasks-vision";
import { pEvent as promisifyEvent } from 'p-event';
import { getImageSegmenter } from './get-image-segmenter.ts';

const imageSegmenter = await getImageSegmenter();

// Get DOM elements
const video = document.getElementById("original") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const overlayCanvas = document.getElementById("overlayCanvas") as HTMLCanvasElement;
const canvasCtx = canvas.getContext("2d", {
  willReadFrequently: true
})!;
const overlayCtx = overlayCanvas.getContext("2d")!;

// Controls
let skinColor = [15, 4, 4];
let skinAlpha = 180;
let hairColor = [200, 87, 51];
let hairAlpha = 128;

const skinColorInput = document.getElementById("skinColor") as HTMLInputElement;
const skinAlphaInput = document.getElementById("skinAlpha") as HTMLInputElement;
const hairColorInput = document.getElementById("hairColor") as HTMLInputElement;
const hairAlphaInput = document.getElementById("hairAlpha") as HTMLInputElement;

skinColorInput.value = rgbToHex(skinColor);
hairColorInput.value = rgbToHex(hairColor);

skinColorInput.onchange = () => skinColor = hexToRgb(skinColorInput.value);
hairColorInput.onchange = () => hairColor = hexToRgb(hairColorInput.value);

skinAlphaInput.value = String(skinAlpha);
hairAlphaInput.value = String(hairAlpha);

skinAlphaInput.onchange = () => skinAlpha = Number(skinAlphaInput.value)
hairAlphaInput.onchange = () => hairAlpha = Number(hairAlphaInput.value);

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
  overlayCanvas.width = video.videoWidth;
  overlayCanvas.height = video.videoHeight;

  console.log(imageSegmenter.getLabels());

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
      let color = [0, 0, 0, 0];

      // If body-skin or face-skin
      if (maskVal === 2 || maskVal === 3) {
        color = skinColor;
        color[3] = skinAlpha;
      }

      // If hair
      if (maskVal === 1) {
        color = hairColor;
        color[3] = hairAlpha;
      }

      canvasImageData[pixelIndex] = color[0];
      canvasImageData[pixelIndex + 1] = color[1];
      canvasImageData[pixelIndex + 2] = color[2];
      canvasImageData[pixelIndex + 3] = color[3];
    }
    const uint8Array = new Uint8ClampedArray(canvasImageData.buffer);
    const dataNew = new ImageData(uint8Array, video.videoWidth, video.videoHeight);

    overlayCtx.putImageData(dataNew, 0, 0);
  }
}

function rgbToHex([r, g, b]: number[]) {
  return "#" + (1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1);
}

function hexToRgb(hex: string) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)!;
  return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)];
}
