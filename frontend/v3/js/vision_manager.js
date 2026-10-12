/**
 * Vision Manager: Orquestador de MediaPipe Pose y MediaPipe Hands
 * Con generador anatómico compuesto para garantizar detección completa de dedos
 * Touchless Projection Mapping V3
 */

export class VisionManager {
  constructor() {
    this.pose = null;
    this.hands = null;
    this.isPoseReady = false;
    this.isHandsReady = false;
    this.isProcessing = false;

    this.onPoseResults = null;
    this.onHandsResults = null;

    this.latestHandLandmarks = [];

    // Canvas compuesto para generar la máscara anatómica completa (brazos + manos + 5 dedos)
    this.maskCanvas = document.createElement("canvas");
    this.maskCanvas.width = 960;
    this.maskCanvas.height = 540;
    this.maskCtx = this.maskCanvas.getContext("2d", { willReadFrequently: false });
  }

  async init(onPoseResults, onHandsResults) {
    this.onPoseResults = onPoseResults;
    this.onHandsResults = onHandsResults;

    // 1. Inicializar MediaPipe Pose con Segmentación Nativa
    if (window.Pose) {
      this.pose = new window.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
      });

      this.pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: true,
        smoothSegmentation: true,
        minDetectionConfidence: 0.55,
        minTrackingConfidence: 0.55
      });

      this.pose.onResults((results) => {
        if (this.onPoseResults) {
          const armData = this._extractArms(results);
          // Generar máscara compuesta que garantiza la inclusión de todos los dedos
          armData.segmentationMask = this._buildCompositeMask(results.segmentationMask, armData);
          this.onPoseResults(results, armData);
        }
      });

      this.isPoseReady = true;
    } else {
      console.error("MediaPipe Pose library not loaded on window.");
    }

    // 2. Inicializar MediaPipe Hands para interacción, gestos y anatomía de dedos
    if (window.Hands) {
      this.hands = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
      });

      this.hands.setOptions({
        modelComplexity: 1,
        maxNumHands: 2,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6
      });

      this.hands.onResults((results) => {
        this.latestHandLandmarks = results.multiHandLandmarks || [];
        if (this.onHandsResults) {
          const gestureData = this._extractPinchGestures(results);
          this.onHandsResults(results, gestureData);
        }
      });

      this.isHandsReady = true;
    } else {
      console.error("MediaPipe Hands library not loaded on window.");
    }
  }

  async processFrame(videoElement) {
    if (this.isProcessing || !videoElement || videoElement.readyState < 2) return;
    this.isProcessing = true;

    try {
      if (this.pose) {
        await this.pose.send({ image: videoElement });
      }
      if (this.hands) {
        await this.hands.send({ image: videoElement });
      }
    } catch (err) {
      console.warn("Frame processing skipped:", err);
    } finally {
      this.isProcessing = false;
    }
  }

  _buildCompositeMask(poseMask, armData) {
    const ctx = this.maskCtx;
    const w = this.maskCanvas.width;
    const h = this.maskCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // 1. Dibujar máscara corporal de Pose (para el torso/antebrazo si existe)
    if (poseMask) {
      ctx.drawImage(poseMask, 0, 0, w, h);
    }

    // 2. Dibujar la anatomía completa de las manos y dedos usando MediaPipe Hands
    // Esto garantiza que NINGÚN dedo se corte, sin importar la resolución de Pose
    if (this.latestHandLandmarks && this.latestHandLandmarks.length > 0) {
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = "#ffffff";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (const hand of this.latestHandLandmarks) {
        if (!hand || hand.length < 21) continue;

        // Calcular escala de la mano (distancia de muñeca a nudillo medio)
        const handScale = Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y);
        const fingerWidth = Math.max(16, Math.min(32, handScale * 0.28 * w));

        // A. Palma de la mano (polígono relleno que une muñeca y nudillos)
        ctx.beginPath();
        ctx.moveTo(hand[0].x * w, hand[0].y * h);
        ctx.lineTo(hand[1].x * w, hand[1].y * h);
        ctx.lineTo(hand[2].x * w, hand[2].y * h);
        ctx.lineTo(hand[5].x * w, hand[5].y * h);
        ctx.lineTo(hand[9].x * w, hand[9].y * h);
        ctx.lineTo(hand[13].x * w, hand[13].y * h);
        ctx.lineTo(hand[17].x * w, hand[17].y * h);
        ctx.closePath();
        ctx.fill();

        // B. Huesos de los 5 dedos con grosor anatómico
        const fingerChains = [
          [0, 1, 2, 3, 4],       // Pulgar
          [0, 5, 6, 7, 8],       // Índice
          [9, 10, 11, 12],       // Medio
          [13, 14, 15, 16],      // Anular
          [0, 17, 18, 19, 20]    // Meñique
        ];

        ctx.lineWidth = fingerWidth;
        for (const chain of fingerChains) {
          ctx.beginPath();
          ctx.moveTo(hand[chain[0]].x * w, hand[chain[0]].y * h);
          for (let i = 1; i < chain.length; i++) {
            ctx.lineTo(hand[chain[i]].x * w, hand[chain[i]].y * h);
          }
          ctx.stroke();

          // Puntas redondeadas en cada yema de dedo
          const tip = hand[chain[chain.length - 1]];
          ctx.beginPath();
          ctx.arc(tip.x * w, tip.y * h, fingerWidth * 0.55, 0, Math.PI * 2);
          ctx.fill();
        }

        // C. Conectar la muñeca con el codo para completar el antebrazo
        const wrist = hand[0];
        let elbow = null;

        // Buscar el codo más cercano detectado por Pose
        if (armData.leftValid && armData.leftArm) {
          elbow = armData.leftArm[1];
        } else if (armData.rightValid && armData.rightArm) {
          elbow = armData.rightArm[1];
        }

        if (elbow) {
          ctx.lineWidth = fingerWidth * 1.6;
          ctx.beginPath();
          ctx.moveTo(wrist.x * w, wrist.y * h);
          ctx.lineTo(elbow.x * w, elbow.y * h);
          ctx.stroke();
        } else {
          // Proyección anatómica hacia abajo/afuera si solo la mano entra en cámara
          const middleKnuckle = hand[9];
          const dx = wrist.x - middleKnuckle.x;
          const dy = wrist.y - middleKnuckle.y;
          ctx.lineWidth = fingerWidth * 1.6;
          ctx.beginPath();
          ctx.moveTo(wrist.x * w, wrist.y * h);
          ctx.lineTo((wrist.x + dx * 1.2) * w, (wrist.y + dy * 1.2) * h);
          ctx.stroke();
        }
      }
    }

    return this.maskCanvas;
  }

  _extractArms(results) {
    const data = {
      leftArm: null,
      rightArm: null,
      leftValid: false,
      rightValid: false,
      segmentationMask: null
    };

    if (!results.poseLandmarks) return data;

    const lm = results.poseLandmarks;
    const minVis = 0.30;

    const leftShoulder = lm[11];
    const leftElbow = lm[13];
    const leftWrist = lm[15];

    if (
      leftShoulder && leftElbow && leftWrist &&
      (leftShoulder.visibility === undefined || leftShoulder.visibility > minVis) &&
      (leftElbow.visibility === undefined || leftElbow.visibility > minVis) &&
      (leftWrist.visibility === undefined || leftWrist.visibility > minVis)
    ) {
      let leftHandTip = null;
      if (lm[19] && (lm[19].visibility === undefined || lm[19].visibility > 0.2)) {
        leftHandTip = { x: lm[19].x, y: lm[19].y };
      } else {
        const dx = leftWrist.x - leftElbow.x;
        const dy = leftWrist.y - leftElbow.y;
        leftHandTip = { x: leftWrist.x + dx * 0.5, y: leftWrist.y + dy * 0.5 };
      }

      data.leftArm = [
        { x: leftShoulder.x, y: leftShoulder.y },
        { x: leftElbow.x, y: leftElbow.y },
        { x: leftWrist.x, y: leftWrist.y },
        leftHandTip
      ];
      data.leftValid = true;
    }

    const rightShoulder = lm[12];
    const rightElbow = lm[14];
    const rightWrist = lm[16];

    if (
      rightShoulder && rightElbow && rightWrist &&
      (rightShoulder.visibility === undefined || rightShoulder.visibility > minVis) &&
      (rightElbow.visibility === undefined || rightElbow.visibility > minVis) &&
      (rightWrist.visibility === undefined || rightWrist.visibility > minVis)
    ) {
      let rightHandTip = null;
      if (lm[20] && (lm[20].visibility === undefined || lm[20].visibility > 0.2)) {
        rightHandTip = { x: lm[20].x, y: lm[20].y };
      } else {
        const dx = rightWrist.x - rightElbow.x;
        const dy = rightWrist.y - rightElbow.y;
        rightHandTip = { x: rightWrist.x + dx * 0.5, y: rightWrist.y + dy * 0.5 };
      }

      data.rightArm = [
        { x: rightShoulder.x, y: rightShoulder.y },
        { x: rightElbow.x, y: rightElbow.y },
        { x: rightWrist.x, y: rightWrist.y },
        rightHandTip
      ];
      data.rightValid = true;
    }

    return data;
  }

  _extractPinchGestures(results) {
    const gestures = [];
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      return gestures;
    }

    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i];
      const indexTip = landmarks[8];
      const thumbTip = landmarks[4];

      const dx = indexTip.x - thumbTip.x;
      const dy = indexTip.y - thumbTip.y;
      const distance = Math.hypot(dx, dy);

      const isPinched = distance < 0.075;

      gestures.push({
        handIndex: i,
        indexTip: { x: indexTip.x, y: indexTip.y },
        thumbTip: { x: thumbTip.x, y: thumbTip.y },
        distance,
        isPinched
      });
    }

    return gestures;
  }
}
