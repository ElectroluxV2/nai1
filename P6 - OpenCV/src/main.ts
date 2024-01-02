import { ImageSegmenterResult } from "@mediapipe/tasks-vision";
import { pEvent as promisifyEvent } from 'p-event';
import { getImageSegmenter } from './get-image-segmenter.ts';
import { legendColors } from './constants.ts';
import cv, { Size, Vector } from '@techstark/opencv-js';

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

  const labels = imageSegmenter.getLabels();
  console.log(labels);

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

    // Draw original webcam data onto canvas
    canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    const startTimeMs = performance.now();

    // Segmenting the stream, single frame at the time
    const result = await new Promise<ImageSegmenterResult>(r => imageSegmenter.segmentForVideo(video, startTimeMs, r));

    // Get original webcam data from main canvas
    const originalImageData = canvasCtx.getImageData(0, 0, video.videoWidth, video.videoHeight);
    const originalImg = cv.matFromImageData(originalImageData);

    // Create face mask onto canvas
    const mask: Float32Array = result.categoryMask!.getAsFloat32Array();
    const canvasImageData = originalImageData.data;

    for (let maskIndex = 0, pixelIndex = 0; maskIndex < mask.length; ++maskIndex, pixelIndex += 4) {
      const maskVal = Math.round(mask[maskIndex] * 255.0);
      const maskColor = maskVal === 3 // Only face skin for now
        ? [255, 255, 255, 255]
        : [0, 0, 0, 255];

      canvasImageData[pixelIndex] = maskColor[0];
      canvasImageData[pixelIndex + 1] = maskColor[1];
      canvasImageData[pixelIndex + 2] = maskColor[2];
      canvasImageData[pixelIndex + 3] = maskColor[3];
    }

    const faceMaskBuffer = new Uint8ClampedArray(canvasImageData.buffer);
    const faceMaskImageData = new ImageData(faceMaskBuffer, video.videoWidth, video.videoHeight);

    // Read face mask as grayscale and threshold to binary
    const faceMaskImg = cv.matFromImageData(faceMaskImageData);
    cv.cvtColor(faceMaskImg, faceMaskImg, cv.COLOR_RGBA2GRAY);
    cv.threshold(faceMaskImg, faceMaskImg, 128, 255, cv.THRESH_BINARY);
    // Antialias mask, convert to float in range 0 to 1 and make 3-channels
    cv.GaussianBlur(faceMaskImg, faceMaskImg, new Size(0, 0), 3, 3, cv.BORDER_DEFAULT);
    cv.equalizeHist(faceMaskImg, faceMaskImg);

    // Get average bgr color of face
    const averageColor = cv.mean(originalImg, faceMaskImg).slice(0, 3);

    // Compute difference colors and make into an image the same size as input
    const desiredColor = [180, 128, 200];
    const differenceColor = [desiredColor[0] - averageColor[0], desiredColor[1] - averageColor[1], desiredColor[2] - averageColor[2]];

    const newImage = new ImageData(new Uint8ClampedArray(originalImageData.data.buffer), originalImageData.width, originalImageData.height);

    // Shift input image color
    for (let pixelIndex = 0; pixelIndex < newImage.data.length; pixelIndex++) {
      newImage.data[pixelIndex] = originalImageData.data[pixelIndex] + differenceColor[0];
      newImage.data[pixelIndex + 1] = originalImageData.data[pixelIndex + 1] + differenceColor[1];
      newImage.data[pixelIndex + 2] = originalImageData.data[pixelIndex + 2] + differenceColor[2];
      newImage.data[pixelIndex + 3] = originalImageData.data[pixelIndex + 3];
    }

    // # combine img and new_img using mask
    // result = (img * (1 - facemask) + new_img * facemask)
    for (let pixelIndex = 0; pixelIndex < newImage.data.length; pixelIndex++) {
      newImage.data[pixelIndex] = originalImageData.data[pixelIndex] * (1 - faceMaskImageData.data[pixelIndex]) + newImage.data[pixelIndex] * faceMaskImageData.data[pixelIndex];
      newImage.data[pixelIndex + 1] = originalImageData.data[pixelIndex + 1] * (1 - faceMaskImageData.data[pixelIndex + 1]) + newImage.data[pixelIndex + 1] * faceMaskImageData.data[pixelIndex + 1];
      newImage.data[pixelIndex + 2] = originalImageData.data[pixelIndex + 2] * (1 - faceMaskImageData.data[pixelIndex + 2]) + newImage.data[pixelIndex + 2] * faceMaskImageData.data[pixelIndex + 2];
      newImage.data[pixelIndex + 3] = originalImageData.data[pixelIndex + 3] * (1 - faceMaskImageData.data[pixelIndex + 3]) + newImage.data[pixelIndex + 3] * faceMaskImageData.data[pixelIndex + 3];
    }

    canvasCtx.putImageData(newImage, 0, 0);

    // cv.merge(new Vector(), faceMaskImg);

    // # combine img and new_img using mask



    // console.log(differenceColor);
    //
    // break;
  }
}
