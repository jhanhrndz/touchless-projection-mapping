/**
 * RVM (Robust Video Matting) Engine
 * Inferencia de matting humano de nivel cinematográfico en tiempo real
 * Touchless Projection Mapping V4
 */

export class RVMEngine {
  constructor(options = {}) {
    this.width = options.width || 512;
    this.height = options.height || 288;
    this.downsampleRatio = options.downsampleRatio || 0.25;

    this.session = null;
    this.isReady = false;
    this.isBusy = false;

    // Estados recurrentes temporales de RVM (inicializados en [1, 1, 1, 1])
    this.r1 = null;
    this.r2 = null;
    this.r3 = null;
    this.r4 = null;
    this.downsampleRatioTensor = null;

    // Buffer planar pre-asignado para evitar recolección de basura continua
    this.numPixels = this.width * this.height;
    this.inputFloatArray = new Float32Array(3 * this.numPixels);

    // Canvas interno para downsampling rápido de la cámara
    this.downsampleCanvas = document.createElement("canvas");
    this.downsampleCanvas.width = this.width;
    this.downsampleCanvas.height = this.height;
    this.downsampleCtx = this.downsampleCanvas.getContext("2d", { willReadFrequently: true });
  }

  async init(modelPath = "models/rvm_mobilenetv3_fp32.onnx") {
    if (!window.ort) {
      throw new Error("onnxruntime-web (ort) no está cargado en el navegador.");
    }

    // Configurar WASM multi-hilos para máximo rendimiento
    window.ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
    window.ort.env.wasm.simd = true;

    try {
      this.session = await window.ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
      console.log("✓ RVM ejecutándose en WebAssembly multi-hilo SIMD (4 núcleos)");
    } catch (e) {
      console.error("Error inicializando sesión RVM:", e);
      throw e;
    }

    this._resetRecurrentStates();
    this.isReady = true;
  }

  _resetRecurrentStates() {
    const ort = window.ort;
    const zero = new Float32Array([0.0]);
    this.r1 = new ort.Tensor("float32", zero, [1, 1, 1, 1]);
    this.r2 = new ort.Tensor("float32", zero, [1, 1, 1, 1]);
    this.r3 = new ort.Tensor("float32", zero, [1, 1, 1, 1]);
    this.r4 = new ort.Tensor("float32", zero, [1, 1, 1, 1]);
    this.downsampleRatioTensor = new ort.Tensor("float32", new Float32Array([this.downsampleRatio]), [1]);
  }

  async processFrame(videoElement) {
    if (!this.isReady || this.isBusy || !videoElement || videoElement.readyState < 2) {
      return null;
    }
    this.isBusy = true;

    try {
      // 1. Extraer frame en resolución de inferencia
      const ctx = this.downsampleCtx;
      const W = this.width;
      const H = this.height;
      ctx.drawImage(videoElement, 0, 0, W, H);
      const imgData = ctx.getImageData(0, 0, W, H).data;

      // 2. Normalizar a formato planar RGB [1, 3, H, W] (0.0 a 1.0)
      const data = this.inputFloatArray;
      const numPixels = this.numPixels;

      for (let i = 0, j = 0; i < numPixels; i++, j += 4) {
        data[i] = imgData[j] / 255.0;                      // R
        data[numPixels + i] = imgData[j + 1] / 255.0;      // G
        data[2 * numPixels + i] = imgData[j + 2] / 255.0;  // B
      }

      const ort = window.ort;
      const srcTensor = new ort.Tensor("float32", data, [1, 3, H, W]);

      const feeds = {
        src: srcTensor,
        r1i: this.r1,
        r2i: this.r2,
        r3i: this.r3,
        r4i: this.r4,
        downsample_ratio: this.downsampleRatioTensor
      };

      // 3. Inferencia RVM
      const results = await this.session.run(feeds);

      // 4. Actualizar memoria recurrente para el siguiente fotograma
      this.r1 = results.r1o;
      this.r2 = results.r2o;
      this.r3 = results.r3o;
      this.r4 = results.r4o;

      // 5. Retornar el mapa alfa (pha)
      return {
        alphaData: results.pha.data, // Float32Array [1, 1, H, W]
        width: W,
        height: H
      };
    } catch (err) {
      console.error("Error en inferencia RVM:", err ? (err.message || err) : "Error desconocido");
      this._resetRecurrentStates();
      return null;
    } finally {
      this.isBusy = false;
    }
  }
}
