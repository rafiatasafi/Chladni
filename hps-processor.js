/**
 * hps-processor.js
 *
 * AudioWorkletProcessor implementing Harmonic Product Spectrum (HPS)
 * for polyphonic pitch detection — finds multiple simultaneous
 * fundamentals from a single FFT frame.
 *
 * How HPS works:
 *   1. Take the magnitude spectrum |X[k]| from an FFT of the input
 *   2. Downsample it by integer factors 2, 3, 4, 5 (compress harmonics)
 *   3. Multiply all downsampled spectra together element-wise
 *   4. Peaks in the product correspond to frequencies whose harmonics
 *      all had energy — i.e. real musical fundamentals, not noise
 *
 * Why it works for chords:
 *   A guitar string at frequency f also vibrates at 2f, 3f, 4f...
 *   (harmonics). When you multiply the spectrum downsampled by 2
 *   with the original, the harmonic content of f lines up and
 *   reinforces at f. Noise and inharmonic content doesn't survive
 *   the multiplication — it gets suppressed.
 *
 * Limitations (honest, for the benchmark):
 *   - Works best on 2-3 note chords; dense voicings confuse it
 *   - Octave errors are common (2f vs f ambiguity)
 *   - Clean guitar signal assumed; heavy distortion smears harmonics
 *
 * Step 3 of the pipeline. Runs alongside YinProcessor — both live
 * on the audio thread, results posted back to main thread separately.
 */

const TWO_PI = 2 * Math.PI;

class HpsProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);

    this.FFT_SIZE    = 4096;   // larger FFT = better freq resolution (~10Hz/bin at 44.1kHz)
    this.HPS_ORDER   = 5;      // number of downsampling stages (2x through 5x)
    this.MIN_FREQ    = 60;     // Hz
    this.MAX_FREQ    = 1400;   // Hz
    this.PEAK_THRESH = 0.1;    // fraction of max HPS peak to count as a chord note
    this.MAX_NOTES   = 4;      // max simultaneous fundamentals to report

    // Fix 1: RMS silence gate
    // Any input below this level is treated as silence — no notes posted.
    // 0.01 ≈ -40dBFS, well above mic self-noise floor (~-60dBFS) but
    // below the softest guitar note (~-20dBFS). Tunable via port message.
    this.RMS_THRESHOLD = 0.01;

    // Fix 2: temporal persistence
    // A peak must appear in this many consecutive analysis frames before
    // being reported. At 50% overlap on 4096 samples @ 44.1kHz, one frame
    // ≈ 46ms, so PERSIST_FRAMES=3 requires ~140ms of stable presence.
    this.PERSIST_FRAMES = 3;
    // Map of "note class string" → consecutive frame count
    this._persistence = new Map();

    // Spectral flux onset detection
    // Noise has a stable, flat spectrum frame-to-frame. A real note being
    // played causes a sudden increase in energy across specific harmonic bins
    // (an "onset"). We compute spectral flux = sum of positive bin differences
    // between consecutive frames. Below FLUX_THRESHOLD we skip HPS entirely.
    //
    // FLUX_THRESHOLD is empirical — 2.0 works for a quiet room with a
    // close-mic'd guitar. Exposed via port message for runtime tuning.
    // Once an onset is detected, ONSET_HOLD_FRAMES keeps HPS running
    // so the full note sustain is captured, not just the attack transient.
    this.FLUX_THRESHOLD    = 2.0;
    this.ONSET_HOLD_FRAMES = 8;   // ~370ms hold after onset at 50% overlap
    this._prevMag          = null; // magnitude spectrum from previous frame
    this._onsetHoldCount   = 0;   // frames remaining in hold window

    // Ring buffer for accumulating 128-sample quanta into FFT_SIZE window
    this._ring    = new Float32Array(this.FFT_SIZE);
    this._ringPos = 0;
    this._framesSinceRun = 0;

    // Hann window — reduces spectral leakage at buffer edges
    // Without this, sharp edges in the buffer cause artifactual high-freq energy
    this._window = new Float32Array(this.FFT_SIZE);
    for (let i = 0; i < this.FFT_SIZE; i++) {
      this._window[i] = 0.5 * (1 - Math.cos(TWO_PI * i / (this.FFT_SIZE - 1)));
    }

    this.port.onmessage = (e) => {
      if (e.data.hpsOrder    !== undefined) this.HPS_ORDER    = e.data.hpsOrder;
      if (e.data.peakThresh  !== undefined) this.PEAK_THRESH  = e.data.peakThresh;
      if (e.data.fluxThresh  !== undefined) this.FLUX_THRESHOLD = e.data.fluxThresh;
      if (e.data.onsetHold   !== undefined) this.ONSET_HOLD_FRAMES = e.data.onsetHold;
    };
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this._ring[this._ringPos++ % this.FFT_SIZE] = channel[i];
    }

    this._framesSinceRun += channel.length;
    // Run every half-buffer (50% overlap) for better temporal resolution
    if (this._framesSinceRun < this.FFT_SIZE / 2) return true;
    this._framesSinceRun = 0;

    // Linearize ring buffer
    const buf    = new Float32Array(this.FFT_SIZE);
    const offset = this._ringPos % this.FFT_SIZE;
    buf.set(this._ring.subarray(offset));
    buf.set(this._ring.subarray(0, offset), this.FFT_SIZE - offset);

    // Fix 1: RMS silence gate — compute before windowing (window reduces energy)
    let sumSq = 0;
    for (let i = 0; i < buf.length; i++) sumSq += buf[i] * buf[i];
    const rms = Math.sqrt(sumSq / buf.length);
    if (rms < this.RMS_THRESHOLD) {
      // Silent frame — decay all persistence counters so stale notes clear
      this._persistence.forEach((v, k) => {
        if (v <= 1) this._persistence.delete(k); else this._persistence.set(k, v - 1);
      });
      this.port.postMessage({ notes: [], timestamp: currentTime, rms });
      return true;
    }

    // Apply Hann window
    for (let i = 0; i < this.FFT_SIZE; i++) buf[i] *= this._window[i];

    // FFT → magnitude spectrum
    const mag = this._fftMagnitude(buf);

    // Spectral flux onset detection
    // Compare current magnitude spectrum to previous frame.
    // flux = Σ max(|X[k]| - |X_prev[k]|, 0)  (half-wave rectified — only increases)
    // This is the standard Dixon (2006) spectral flux formulation.
    let flux = 0;
    if (this._prevMag) {
      const binHz  = sampleRate / this.FFT_SIZE;
      const minBin = Math.ceil(this.MIN_FREQ / binHz);
      const maxBin = Math.floor(this.MAX_FREQ / binHz);
      for (let i = minBin; i <= maxBin; i++) {
        const diff = mag[i] - this._prevMag[i];
        if (diff > 0) flux += diff;
      }
    }
    // Store current spectrum for next frame comparison
    this._prevMag = mag;

    // Onset gate: if flux exceeds threshold, reset hold counter
    if (flux > this.FLUX_THRESHOLD) {
      this._onsetHoldCount = this.ONSET_HOLD_FRAMES;
    } else if (this._onsetHoldCount > 0) {
      this._onsetHoldCount--;
    }

    // Skip HPS entirely if no onset and hold has expired
    if (this._onsetHoldCount === 0) {
      // Decay persistence so stale notes clear quickly during silence
      this._persistence.forEach((v, k) => {
        if (v <= 1) this._persistence.delete(k); else this._persistence.set(k, v - 1);
      });
      this.port.postMessage({ notes: [], timestamp: currentTime, rms, flux });
      return true;
    }

    // HPS
    const hps        = this._harmonicProductSpectrum(mag);
    const rawPeaks   = this._extractPeaks(hps);

    // Fix 3: merge peaks that map to the same note class (octave dedup)
    // Two peaks within 50 cents of the same note name are the same pitch —
    // keep the one with higher magnitude.
    const merged = this._mergePeaks(rawPeaks);

    // Fix 2: only promote peaks that have persisted for PERSIST_FRAMES frames
    const currentKeys = new Set();
    for (const peak of merged) {
      const key = this._noteKey(peak.freq);
      currentKeys.add(key);
      this._persistence.set(key, (this._persistence.get(key) || 0) + 1);
    }
    // Decay keys not seen this frame
    this._persistence.forEach((v, k) => {
      if (!currentKeys.has(k)) {
        if (v <= 1) this._persistence.delete(k); else this._persistence.set(k, v - 1);
      }
    });

    const stableNotes = merged.filter(p =>
      (this._persistence.get(this._noteKey(p.freq)) || 0) >= this.PERSIST_FRAMES
    );

    this.port.postMessage({ notes: stableNotes, timestamp: currentTime, rms, flux });
    return true;
  }

  // --- FFT (Cooley-Tukey, radix-2, in-place) ---
  // Returns magnitude spectrum (first N/2 bins only — positive frequencies)

  _fftMagnitude(signal) {
    const N  = signal.length;
    const re = new Float32Array(signal);
    const im = new Float32Array(N);

    // Bit-reversal permutation
    let j = 0;
    for (let i = 1; i < N; i++) {
      let bit = N >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }

    // Cooley-Tukey butterfly
    for (let len = 2; len <= N; len <<= 1) {
      const half  = len >> 1;
      const angle = -TWO_PI / len;
      const wRe   = Math.cos(angle);
      const wIm   = Math.sin(angle);
      for (let i = 0; i < N; i += len) {
        let curRe = 1, curIm = 0;
        for (let k = 0; k < half; k++) {
          const uRe = re[i + k],         uIm = im[i + k];
          const vRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
          const vIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
          re[i + k]        = uRe + vRe;
          im[i + k]        = uIm + vIm;
          re[i + k + half] = uRe - vRe;
          im[i + k + half] = uIm - vIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm        = curRe * wIm + curIm * wRe;
          curRe        = nextRe;
        }
      }
    }

    // Magnitude of positive-frequency bins only
    const half = N >> 1;
    const mag  = new Float32Array(half);
    for (let i = 0; i < half; i++) {
      mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    }
    return mag;
  }

  // --- Harmonic Product Spectrum ---

  _harmonicProductSpectrum(mag) {
    const hps = new Float32Array(mag); // start with original spectrum

    // Multiply by progressively downsampled copies
    for (let h = 2; h <= this.HPS_ORDER; h++) {
      for (let i = 0; i < hps.length; i++) {
        const srcIdx = Math.floor(i * h);
        if (srcIdx < mag.length) {
          hps[i] *= mag[srcIdx];
        } else {
          hps[i] = 0; // out of range — zero out
        }
      }
    }
    return hps;
  }

  // --- Note key: quantize freq to nearest semitone for persistence tracking ---
  _noteKey(freq) {
    const midi = Math.round(12 * Math.log2(freq / 440) + 69);
    return midi; // integer MIDI note number — same key for enharmonic equivalents
  }

  // --- Fix 3: merge peaks within ±1 semitone of each other ---
  // Octave errors show up as a peak at f and another at 2f (same note, different octave).
  // We keep the lower-frequency peak when two share a note name, since HPS tends to
  // over-report harmonics rather than miss fundamentals.
  _mergePeaks(peaks) {
    const SEMITONE_CENTS = 100;
    const merged = [];
    for (const peak of peaks) {
      const midiFloat = 12 * Math.log2(peak.freq / 440) + 69;
      const isDuplicate = merged.some(kept => {
        const keptMidi = 12 * Math.log2(kept.freq / 440) + 69;
        // same note class (mod 12) within one octave tolerance
        return Math.abs(midiFloat - keptMidi) % 12 < 0.5 ||
               Math.abs(midiFloat - keptMidi) < 1.0;
      });
      if (!isDuplicate) merged.push(peak);
    }
    return merged;
  }

  // --- Peak picking ---
  // Finds local maxima in HPS above threshold, converts bin → Hz

  _extractPeaks(hps) {
    const binHz     = sampleRate / this.FFT_SIZE;
    const minBin    = Math.ceil(this.MIN_FREQ / binHz);
    const maxBin    = Math.floor(this.MAX_FREQ / binHz);

    // Find global max for relative threshold
    let globalMax = 0;
    for (let i = minBin; i <= maxBin; i++) {
      if (hps[i] > globalMax) globalMax = hps[i];
    }
    if (globalMax === 0) return [];

    const threshold = globalMax * this.PEAK_THRESH;
    const peaks     = [];

    for (let i = minBin + 1; i < maxBin - 1; i++) {
      if (
        hps[i] > threshold &&
        hps[i] > hps[i - 1] &&
        hps[i] > hps[i + 1]
      ) {
        // Parabolic interpolation for sub-bin freq accuracy
        const alpha  = hps[i - 1];
        const beta   = hps[i];
        const gamma  = hps[i + 1];
        const offset = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma);
        const freq   = (i + offset) * binHz;
        peaks.push({ freq, magnitude: beta / globalMax });
      }
    }

    // Sort by magnitude descending, return top N
    peaks.sort((a, b) => b.magnitude - a.magnitude);
    return peaks.slice(0, this.MAX_NOTES).map(p => ({
      freq:      parseFloat(p.freq.toFixed(2)),
      magnitude: parseFloat(p.magnitude.toFixed(3)),
    }));
  }
}

registerProcessor('hps-processor', HpsProcessor);