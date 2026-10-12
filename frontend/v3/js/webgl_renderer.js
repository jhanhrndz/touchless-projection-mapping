/**
 * WebGL 2.0 Hardware Shader Renderer para Segmentación y Efectos de Brazo
 * Touchless Projection Mapping V3
 */

export class WebGLArmRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", {
      alpha: true,
      premultipliedAlpha: false,
      antialias: true,
      preserveDrawingBuffer: false
    });

    if (!this.gl) {
      throw new Error("WebGL 2.0 no soportado en este navegador.");
    }

    this.program = null;
    this.uniforms = {};
    this.cameraTexture = null;
    this.maskTexture = null;
    this.quadVAO = null;

    this._initShaders();
    this._initBuffers();
    this._initTextures();
  }

  _initShaders() {
    const gl = this.gl;

    const vsSource = `#version 300 es
    in vec2 a_position;
    out vec2 v_uv;

    void main() {
      // Mapear de [-1, 1] a [0, 1]
      v_uv = (a_position + 1.0) * 0.5;
      v_uv.y = 1.0 - v_uv.y; // Invertir Y para coordenadas estándar
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
    `;

    const fsSource = `#version 300 es
    precision highp float;

    in vec2 v_uv;
    out vec4 fragColor;

    uniform sampler2D u_camera_tex;
    uniform sampler2D u_mask_tex;

    uniform vec2 u_left_arm[4];   // 0: hombro, 1: codo, 2: muñeca, 3: punta de dedos
    uniform vec2 u_right_arm[4];  // 0: hombro, 1: codo, 2: muñeca, 3: punta de dedos
    uniform float u_left_valid;
    uniform float u_right_valid;
    uniform float u_arm_radius;
    uniform float u_time;
    uniform int u_effect_mode;
    uniform float u_mirror;

    // Distancia euclidiana mínima de un punto P al segmento de recta AB
    float distToSegment(vec2 p, vec2 a, vec2 b) {
      vec2 pa = p - a;
      vec2 ba = b - a;
      float d = dot(ba, ba);
      if (d <= 0.00001) return length(pa);
      float h = clamp(dot(pa, ba) / d, 0.0, 1.0);
      return length(pa - ba * h);
    }

    // Muestreo bilineal multicapa con anti-aliasing (elimina el pixelado de la máscara 256x256)
    float sampleSmoothMask(vec2 coords) {
      vec2 texel = 1.0 / vec2(textureSize(u_mask_tex, 0));
      float c = texture(u_mask_tex, coords).r * 0.40;
      float l = texture(u_mask_tex, coords - vec2(texel.x, 0.0)).r * 0.15;
      float r = texture(u_mask_tex, coords + vec2(texel.x, 0.0)).r * 0.15;
      float u = texture(u_mask_tex, coords - vec2(0.0, texel.y)).r * 0.15;
      float d = texture(u_mask_tex, coords + vec2(0.0, texel.y)).r * 0.15;
      float sum = c + l + r + u + d;
      return smoothstep(0.28, 0.68, sum);
    }

    void main() {
      vec2 uv = v_uv;
      if (u_mirror > 0.5) {
        uv.x = 1.0 - uv.x;
      }

      // 1. Distancia mínima a los segmentos anatómicos
      float minDist = 999.0;

      if (u_left_valid > 0.5) {
        float d_bicep = distToSegment(uv, u_left_arm[0], u_left_arm[1]);
        float d_forearm = distToSegment(uv, u_left_arm[1], u_left_arm[2]);
        // Factor 0.55 en d_hand permite que los 5 dedos abiertos queden 100% cubiertos
        float d_hand = distToSegment(uv, u_left_arm[2], u_left_arm[3]) * 0.55;
        minDist = min(minDist, min(d_bicep, min(d_forearm, d_hand)));
      }

      if (u_right_valid > 0.5) {
        float d_bicep = distToSegment(uv, u_right_arm[0], u_right_arm[1]);
        float d_forearm = distToSegment(uv, u_right_arm[1], u_right_arm[2]);
        float d_hand = distToSegment(uv, u_right_arm[2], u_right_arm[3]) * 0.55;
        minDist = min(minDist, min(d_bicep, min(d_forearm, d_hand)));
      }

      // Si no hay brazos detectados, descartar
      if (u_left_valid < 0.5 && u_right_valid < 0.5) {
        fragColor = vec4(0.0);
        return;
      }

      // 2. Máscara de Segmentación con Anti-Aliasing Suave
      float humanMask = sampleSmoothMask(uv);

      // 3. Máscara de Cápsula con caída suave
      float capsuleMask = 1.0 - smoothstep(u_arm_radius * 0.70, u_arm_radius * 1.30, minDist);

      // 4. Combinación: Píxeles del brazo y mano
      float armAlpha = humanMask * capsuleMask;

      if (armAlpha < 0.02) {
        fragColor = vec4(0.0);
        return;
      }

      // 5. Textura de Cámara y Detección de Circuitos Digitales
      vec4 cam = texture(u_camera_tex, uv);
      float lum = dot(cam.rgb, vec3(0.299, 0.587, 0.114));

      // Detección de derivadas por hardware: transforma nudillos, pliegues y bordes reales en circuitos digitales brillantes
      float circuits = clamp(length(vec2(dFdx(lum), dFdy(lum))) * 14.0, 0.0, 1.0);

      // Aura perimetral en la silueta externa
      float rimGlow = smoothstep(u_arm_radius * 0.40, u_arm_radius * 1.05, minDist);

      vec3 finalColor = vec3(0.0);

      if (u_effect_mode == 0) {
        // MATRIX CYBERNETIC (Verde Neón Holográfico - Estilo Cyberpunk / Tron)
        // Núcleo translúcido digital
        vec3 core = vec3(0.05, 0.90, 0.35) * (lum * 0.65 + 0.35);
        // Circuitos de datos que iluminan los dedos reales
        vec3 cyberLines = vec3(0.60, 1.0, 0.80) * circuits * 2.5;
        // Borde neón brillante
        vec3 rim = vec3(0.10, 1.0, 0.45) * rimGlow * 2.8;
        // Scanlines animadas
        float scan = sin(uv.y * 380.0 - u_time * 8.0) * 0.12 + 0.88;
        finalColor = (core + cyberLines + rim) * scan;
      } 
      else if (u_effect_mode == 1) {
        // CYBERPUNK CYAN DIGITAL (Cian eléctrico con circuitos de alta luminosidad)
        vec3 core = vec3(0.0, 0.70, 1.0) * (lum * 0.7 + 0.3);
        vec3 cyberLines = vec3(0.70, 0.95, 1.0) * circuits * 2.8;
        vec3 rim = vec3(0.10, 0.95, 1.0) * rimGlow * 3.2;
        float scan = sin(uv.y * 260.0 + u_time * 6.0) * 0.14 + 0.86;
        finalColor = (core + cyberLines + rim) * scan;
      } 
      else if (u_effect_mode == 2) {
        // PLASMA HOLOGRÁFICO PULSANTE (Fluctuación de energía pura)
        float pulse = sin(u_time * 4.5) * 0.15 + 0.85;
        vec3 core = vec3(0.15, 0.85, 1.0) * (lum * 0.8 + 0.2) * pulse;
        vec3 cyberLines = vec3(0.6, 1.0, 0.9) * circuits * 2.2;
        vec3 rim = vec3(0.3, 0.85, 1.0) * rimGlow * 2.6;
        finalColor = core + cyberLines + rim;
      } 
      else {
        // SILUETA DIGITAL SÓLIDA INTENSA (Verde puro Matrix)
        vec3 neon = vec3(0.05, 1.0, 0.30);
        vec3 cyberLines = vec3(0.8, 1.0, 0.9) * circuits * 1.8;
        vec3 rim = vec3(0.2, 1.0, 0.5) * rimGlow * 2.2;
        finalColor = neon * 0.75 + cyberLines + rim;
      }

      // Salida con alpha premultiplicado para proyección aditiva perfecta
      fragColor = vec4(finalColor * armAlpha, armAlpha);
    }
    `;

    const vs = this._compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSource);

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Error enlazando Shader Program: ${gl.getProgramInfoLog(program)}`);
    }

    this.program = program;

    // Cachear Uniforms
    const uniformNames = [
      "u_camera_tex", "u_mask_tex",
      "u_left_arm", "u_right_arm",
      "u_left_valid", "u_right_valid",
      "u_arm_radius", "u_time",
      "u_effect_mode", "u_mirror"
    ];
    for (const name of uniformNames) {
      this.uniforms[name] = gl.getUniformLocation(program, name);
    }
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Fallo compilando Shader (${type}): ${info}`);
    }
    return shader;
  }

  _initBuffers() {
    const gl = this.gl;
    // Cuadrilátero que cubre toda la pantalla (-1 a 1)
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1,
    ]);

    this.quadVAO = gl.createVertexArray();
    gl.bindVertexArray(this.quadVAO);

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const posAttr = gl.getAttribLocation(this.program, "a_position");
    gl.enableVertexAttribArray(posAttr);
    gl.vertexAttribPointer(posAttr, 2, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);
  }

  _initTextures() {
    const gl = this.gl;

    // Textura 0: Cámara
    this.cameraTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cameraTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Textura 1: Máscara de Segmentación
    this.maskTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  }

  updateCameraTexture(video) {
    if (!video || video.readyState < 2) return;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cameraTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
  }

  updateMaskTexture(mask) {
    if (!mask) return;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, mask);
  }

  render(params) {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);

    gl.uniform1i(this.uniforms.u_camera_tex, 0);
    gl.uniform1i(this.uniforms.u_mask_tex, 1);

    // Brazos: Hombro, Codo, Muñeca, Mano/Dedos (4 puntos anatómicos)
    if (params.leftArm && params.leftArm.length >= 3) {
      const p3 = params.leftArm[3] || params.leftArm[2];
      const leftCoords = new Float32Array([
        params.leftArm[0].x, params.leftArm[0].y,
        params.leftArm[1].x, params.leftArm[1].y,
        params.leftArm[2].x, params.leftArm[2].y,
        p3.x, p3.y
      ]);
      gl.uniform2fv(this.uniforms.u_left_arm, leftCoords);
      gl.uniform1f(this.uniforms.u_left_valid, 1.0);
    } else {
      gl.uniform1f(this.uniforms.u_left_valid, 0.0);
    }

    if (params.rightArm && params.rightArm.length >= 3) {
      const p3 = params.rightArm[3] || params.rightArm[2];
      const rightCoords = new Float32Array([
        params.rightArm[0].x, params.rightArm[0].y,
        params.rightArm[1].x, params.rightArm[1].y,
        params.rightArm[2].x, params.rightArm[2].y,
        p3.x, p3.y
      ]);
      gl.uniform2fv(this.uniforms.u_right_arm, rightCoords);
      gl.uniform1f(this.uniforms.u_right_valid, 1.0);
    } else {
      gl.uniform1f(this.uniforms.u_right_valid, 0.0);
    }

    gl.uniform1f(this.uniforms.u_arm_radius, params.armRadius || 0.11);
    gl.uniform1f(this.uniforms.u_time, params.time || 0.0);
    gl.uniform1i(this.uniforms.u_effect_mode, params.effectMode !== undefined ? params.effectMode : 0);
    gl.uniform1f(this.uniforms.u_mirror, params.mirror ? 1.0 : 0.0);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }
}
