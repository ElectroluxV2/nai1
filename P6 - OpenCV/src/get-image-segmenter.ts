import { ImageSegmenter } from '@mediapipe/tasks-vision';
import wasmLoaderPath from '../node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal.js?url';
import wasmBinaryPath from '../node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm?url';

export const getImageSegmenter = async () => {
  const wasmFileset: Parameters<typeof ImageSegmenter.createFromOptions>[0] = {
    wasmBinaryPath: wasmBinaryPath,
    wasmLoaderPath: wasmLoaderPath
  };

  return await ImageSegmenter.createFromOptions(wasmFileset, {
    baseOptions: {
      modelAssetPath: "/selfie_multiclass_256x256.tflite",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
}
