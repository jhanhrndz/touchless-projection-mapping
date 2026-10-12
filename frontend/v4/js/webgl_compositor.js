/**
 * WebGL 2.0 Compositor para RVM (Robust Video Matting) y Shaders de Brazo
 * Touchless Projection Mapping V4
 */

export class WebGLCompositor {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", {
      alpha: true,
      premultipliedAlpha: false,
      antialias: true,
      preserveDrawingBuffer: false
    });

    if (!this.gl) {
      throw new Error("WebGL 2.0 no soportado.");
    }

    this.program = null;
    this.uniforms = {};
    this.cameraTexture = null;
    this.rvmTexture = null;
    this.quadVAO = null;

    this.maskUint8 = null;

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
      v_uv = (a_position + 1.0) * 0.5;
      v_uv.y = 1.0 - v_uv.y;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
    `;

    const fsSource = `#version 300 es
    precision highp float;

    in vec2 v_uv;
    out vec4 fragColor;

    uniform sampler2D u_camera_tex;
    uniform sampler2D u_rvm_mask;

    uniform vec2 u_left_arm[4];   // 0: hombro, 1: codo, 2: muñeca, 3: punta de mano
    uniform vec2 u_right_arm[4];
    uniform float u_left_valid;
    uniform float u_right_valid;
    uniform float u_arm_radius;
    uniform float u_time;
    uniform int u_effect_mode;
    uniform float u_mirror;
    uniform float u_filter_arms;

    float distToSegment(vec2 p, vec2 a, vec2 b) {
      vec2 aspect = vec2(1.7777778, 1.0);
      vec2 pa = (p - a) * aspect;
      vec2 ba = (b - a) * aspect;
      float d = dot(ba, ba);
      if (d <= 0.00001) return length(pa);
      float h = clamp(dot(pa, ba) / d, 0.0, 1.0);
      return length(pa - ba * h);
    }

    void main() {
      vec2 uv = v_uv;
      if (u_mirror > 0.5) {
        uv.x = 1.0 - uv.x;
      }

      // 1. Distancia a las cápsulas anatómicas de los brazos
      float minDist = 999.0;

      if (u_left_valid > 0.5) {
        // Cortar segmento superior para que termine en el brazo y nunca toque la clavícula, cuello o torso
        vec2 leftUpperCut = mix(u_left_arm[1], u_left_arm[0], 0.60);
        float d1 = distToSegment(uv, leftUpperCut, u_left_arm[1]);
        float d2 = distToSegment(uv, u_left_arm[1], u_left_arm[2]);
        float d3 = distToSegment(uv, u_left_arm[2], u_left_arm[3]) * 0.60;
        minDist = min(minDist, min(d1, min(d2, d3)));
      }

      if (u_right_valid > 0.5) {
        vec2 rightUpperCut = mix(u_right_arm[1], u_right_arm[0], 0.60);
        float d1 = distToSegment(uv, rightUpperCut, u_right_arm[1]);
        float d2 = distToSegment(uv, u_right_arm[1], u_right_arm[2]);
        float d3 = distToSegment(uv, u_right_arm[2], u_right_arm[3]) * 0.60;
        minDist = min(minDist, min(d1, min(d2, d3)));
      }

      // 2. Leer Alpha Matte fotográfico de RVM (0.0 fondo a 1.0 humano con precisión subpíxel)
      float rvmAlpha = texture(u_rvm_mask, uv).r;

      // 3. Máscara de filtro de brazo (transición limpia y precisa sin invadir el torso)
      float capsuleMask = 1.0 - smoothstep(u_arm_radius * 0.85, u_arm_radius * 1.15, minDist);

      // Si u_filter_arms está activo: aislar únicamente los brazos si hay alguno detectado.
      // Si u_filter_arms no está activo: mostrar la silueta RVM completa.
      float armAlpha = 0.0;
      if (u_filter_arms > 0.5) {
        if (u_left_valid > 0.5 || u_right_valid > 0.5) {
          armAlpha = rvmAlpha * capsuleMask;
        } else {
          armAlpha = 0.0;
        }
      } else {
        armAlpha = rvmAlpha;
      }

      if (armAlpha < 0.02) {
        fragColor = vec4(0.0);
        return;
      }

      // 4. Imagen de Cámara y Detección de Circuitos Digitales
      vec4 cam = texture(u_camera_tex, uv);
      float lum = dot(cam.rgb, vec3(0.299, 0.587, 0.114));

      // Detección de derivadas parciales: resalta nudillos, uñas, pliegues de piel como circuitos brillantes
      float circuits = clamp(length(vec2(dFdx(lum), dFdy(lum))) * 14.0, 0.0, 1.0);

      // Resplandor perimetral en la silueta exacta de RVM
      vec2 texel = 1.0 / vec2(textureSize(u_rvm_mask, 0));
      float rvmNeighbor = (
        texture(u_rvm_mask, uv + vec2(texel.x * 2.0, 0.0)).r +
        texture(u_rvm_mask, uv - vec2(texel.x * 2.0, 0.0)).r +
        texture(u_rvm_mask, uv + vec2(0.0, texel.y * 2.0)).r +
        texture(u_rvm_mask, uv - vec2(0.0, texel.y * 2.0)).r
      ) * 0.25;
      float rimGlow = smoothstep(0.1, 0.9, rvmAlpha) * (1.0 - smoothstep(0.65, 0.98, rvmNeighbor));

      vec3 finalColor = vec3(0.0);

      if (u_effect_mode == 0) {
        // MODO 0: CYBER-ARM REALISTA (Piel Real de la cámara + Circuitos Neón HUD + Rim Glow)
        // Mantiene 100% el realismo de la piel y superpone circuitos biónicos y brillo perimetral
        vec3 cyberLines = vec3(0.25, 1.0, 0.65) * circuits * 3.2;
        vec3 rim = vec3(0.1, 0.95, 1.0) * rimGlow * 3.6;
        float scan = sin(uv.y * 360.0 - u_time * 6.0) * 0.06 + 0.94;
        vec3 skinHolo = cam.rgb * 1.2 + cyberLines + rim;
        finalColor = skinHolo * scan;
      }
      else if (u_effect_mode == 1) {
        // MODO 1: MATRIX RVM (Verde Neón Fotográfico + Circuitos Vivos)
        vec3 core = vec3(0.05, 0.90, 0.35) * (lum * 0.7 + 0.3);
        vec3 cyberLines = vec3(0.60, 1.0, 0.80) * circuits * 2.6;
        vec3 rim = vec3(0.15, 1.0, 0.45) * rimGlow * 3.5;
        float scan = sin(uv.y * 420.0 - u_time * 8.0) * 0.12 + 0.88;
        finalColor = (core + cyberLines + rim) * scan;
      } 
      else if (u_effect_mode == 2) {
        // MODO 2: CYBERPUNK CYAN RVM (Cian eléctrico con aura de borde)
        vec3 core = vec3(0.0, 0.70, 1.0) * (lum * 0.75 + 0.25);
        vec3 cyberLines = vec3(0.70, 0.95, 1.0) * circuits * 2.8;
        vec3 rim = vec3(0.10, 0.95, 1.0) * rimGlow * 3.8;
        float scan = sin(uv.y * 280.0 + u_time * 6.0) * 0.14 + 0.86;
        finalColor = (core + cyberLines + rim) * scan;
      } 
      else if (u_effect_mode == 3) {
        // MODO 3: PLASMA HOLOGRÁFICO PULSANTE
        float pulse = sin(u_time * 4.5) * 0.15 + 0.85;
        vec3 core = vec3(0.15, 0.85, 1.0) * (lum * 0.8 + 0.2) * pulse;
        vec3 cyberLines = vec3(0.6, 1.0, 0.9) * circuits * 2.2;
        vec3 rim = vec3(0.3, 0.85, 1.0) * rimGlow * 3.0;
        finalColor = core + cyberLines + rim;
      } 
      else {
        // MODO 4: SILUETA DIGITAL SÓLIDA MATRIX
        vec3 neon = vec3(0.05, 1.0, 0.30);
        vec3 cyberLines = vec3(0.8, 1.0, 0.9) * circuits * 1.8;
        vec3 rim = vec3(0.2, 1.0, 0.5) * rimGlow * 2.5;
        finalColor = neon * 0.75 + cyberLines + rim;
      }

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
      throw new Error(`Error enlazando Shader Compositor: ${gl.getProgramInfoLog(program)}`);
    }

    this.program = program;

    const uniformNames = [
      "u_camera_tex", "u_rvm_mask",
      "u_left_arm", "u_right_arm",
      "u_left_valid", "u_right_valid",
      "u_arm_radius", "u_time",
      "u_effect_mode", "u_mirror",
      "u_filter_arms"
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
      throw new Error(`Fallo compilando Shader: ${info}`);
    }
    return shader;
  }

  _initBuffers() {
    const gl = this.gl;
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

    // Textura de Cámara
    this.cameraTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cameraTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Textura RVM Alpha (R8)
    this.rvmTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.rvmTexture);
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

  updateRvmMask(alphaFloatData, width, height) {
    if (!alphaFloatData) return;
    const gl = this.gl;

    const len = alphaFloatData.length;
    if (!this.maskUint8 || this.maskUint8.length !== len) {
      this.maskUint8 = new Uint8Array(len);
    }

    const u8 = this.maskUint8;
    for (let i = 0; i < len; i++) {
      u8[i] = Math.min(255, Math.max(0, alphaFloatData[i] * 255.0));
    }

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.rvmTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, u8);
  }

  render(params) {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);

    gl.uniform1i(this.uniforms.u_camera_tex, 0);
    gl.uniform1i(this.uniforms.u_rvm_mask, 1);

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

    gl.uniform1f(this.uniforms.u_arm_radius, params.armRadius || 0.12);
    gl.uniform1f(this.uniforms.u_time, params.time || 0.0);
    gl.uniform1i(this.uniforms.u_effect_mode, params.effectMode !== undefined ? params.effectMode : 0);
    gl.uniform1f(this.uniforms.u_mirror, params.mirror ? 1.0 : 0.0);
    gl.uniform1f(this.uniforms.u_filter_arms, params.filterArms !== undefined ? (params.filterArms ? 1.0 : 0.0) : 1.0);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }
}
