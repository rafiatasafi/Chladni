/**
 * yin-processor.js
 *
 * AudioWorkletProcessor that runs YIN pitch detection on the audio
 * rendering thread — no main-thread involvement during detection,
 * no dropped frames, no UI jank.
 *
 * CRITICAL: this file runs in the AudioWorkletGlobalScope, which is
 * a separate JS environment from the main window. It has NO access to:
 *   - the DOM
 *   - window / document
 *   - import() or ES module imports  ← this is why YIN is inlined here
 *
 * Communication back to the main thread happens exclusively via
 * this.port.postMessage(), which uses the structured clone algorithm
 * (fast, zero-copy for transferable objects like Float32Array).
 *
 * Audio rendering thread guarantees:
 *   - process() is called every 128 samples (~2.9ms at 44.1kHz)
 *   - the browser scheduler gives this thread hard real-time priority
 *   - any blocking in process() causes audible glitches ("audio dropout")
 *   - YIN on a 2048-sample buffer takes ~0.5ms on modern hardware — safe
 *
 * Step 2 of the Chord Quest DSP upgrade. Step 3 will add HPS-based
 * polyphonic detection in a second processor (or an extended version
 * of this one that also runs an FFT).
 */

class YinProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);

    /**
     * Internal ring buffer: we accumulate 128-sample quanta from the
     * Web Audio pipeline until we have BUFFER_SIZE samples, then run
     * YIN on the full buffer.  Running on every 128-sample frame would
     * produce ~344 detections/second — wasteful and noisier than running
     * on a longer window.
     */
    this.BUFFER_SIZE = 2048;    // ~46ms at 44.1kHz — same as main-thread version
    this.THRESHOLD   = 0.15;    // YIN confidence threshold (lower = stricter)
    this.MIN_FREQ    = 60;      // Hz — below low E on guitar
    this.MAX_FREQ    = 1400;    // Hz — above high E, 12th fret

    this._ring    = new Float32Array(this.BUFFER_SIZE);
    this._ringPos = 0;
    this._framesSincePost = 0;

    // Receive config updates from main thread (e.g. threshold tweaks)
    this.port.onmessage = (e) => {
      if (e.data.threshold !== undefined) this.THRESHOLD = e.data.threshold;
    };
  }

  /**
   * process() is called by the audio rendering thread every 128 samples.
   * inputs[0][0] is the first channel of the first input (mono mic feed).
   * Return true to keep the node alive; false to let it be GC'd.
   */
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    // Fill ring buffer with this frame's samples
    for (let i = 0; i < channel.length; i++) {
      this._ring[this._ringPos++ % this.BUFFER_SIZE] = channel[i];
    }

    // Only run YIN and post a result every full buffer's worth of frames
    this._framesSincePost += channel.length;
    if (this._framesSincePost < this.BUFFER_SIZE) return true;
    this._framesSincePost = 0;

    // Linearize the ring buffer into a contiguous snapshot for YIN
    const snapshot = new Float32Array(this.BUFFER_SIZE);
    const offset   = this._ringPos % this.BUFFER_SIZE;
    snapshot.set(this._ring.subarray(offset));
    snapshot.set(this._ring.subarray(0, offset), this.BUFFER_SIZE - offset);

    const freq = this._yinDetect(snapshot);

    // Post result regardless — null means "silence/noise this frame"
    // The main thread decides how to display it (hold last note, dim, etc.)
    this.port.postMessage({ freq, timestamp: currentTime });

    return true; // keep alive
  }

  // --- YIN algorithm (inlined — no imports allowed in worklet scope) ---

  _yinDetect(buffer) {
    const half = Math.floor(buffer.length / 2);
    const yin  = new Float32Array(half);

    // Step 1: difference function
    for (let tau = 1; tau < half; tau++) {
      let sum = 0;
      for (let i = 0; i < half; i++) {
        const d = buffer[i] - buffer[i + tau];
        sum += d * d;
      }
      yin[tau] = sum;
    }

    // Step 2: cumulative mean normalised difference
    yin[0] = 1;
    let running = 0;
    for (let tau = 1; tau < half; tau++) {
      running += yin[tau];
      yin[tau] *= tau / running;
    }

    // Step 3: absolute threshold + local minimum
    let tau = -1;
    for (let t = 2; t < half; t++) {
      if (yin[t] < this.THRESHOLD) {
        while (t + 1 < half && yin[t + 1] < yin[t]) t++;
        tau = t;
        break;
      }
    }
    if (tau === -1) return null;

    // Step 4: parabolic interpolation (sub-sample precision)
    const x0 = tau > 0       ? tau - 1 : tau;
    const x2 = tau + 1 < half ? tau + 1 : tau;
    let refined;
    if (x0 === tau) {
      refined = yin[tau] <= yin[x2] ? tau : x2;
    } else if (x2 === tau) {
      refined = yin[tau] <= yin[x0] ? tau : x0;
    } else {
      const s0 = yin[x0], s1 = yin[tau], s2 = yin[x2];
      refined = tau + (s2 - s0) / (2 * (2 * s1 - s2 - s0));
    }

    if (refined <= 0) return null;
    const freq = sampleRate / refined; // sampleRate is a global in AudioWorkletGlobalScope
    return freq >= this.MIN_FREQ && freq <= this.MAX_FREQ ? freq : null;
  }
}

registerProcessor('yin-processor', YinProcessor);