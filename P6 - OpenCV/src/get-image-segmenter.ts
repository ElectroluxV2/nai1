import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

export const getImageSegmenter = async () => {
  const wasmAssets = import.meta.glob('/node_modules/@mediapipe/tasks-vision/wasm/*', {
    as: 'url',
    eager: true,
  });

  const firstWasmModulePath = wasmAssets[Object.keys(wasmAssets)[0]];
  const wasmModulesBasePath = firstWasmModulePath.substring(0, firstWasmModulePath.lastIndexOf('/'));
  const wasmFileset = await FilesetResolver.forVisionTasks(wasmModulesBasePath);

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
