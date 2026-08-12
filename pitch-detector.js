/**
 * pitch-detector.js
 *
 * Single-note fundamental frequency detection using the YIN algorithm
 * (de Cheveigné & Kawahara, 2002). This is step 1 of the Chord Quest
 * DSP upgrade: main-thread version for fast iteration. Step 2 moves
 * this same core loop into an AudioWorkletProcessor so it runs on
 * the audio rendering thread instead of blocking on rAF.
 *
 * Usage:
 *   import { yinPitchDetect, frequencyToNote } from './pitch-detector.js';
 *   const freq = yinPitchDetect(timeDomainBuffer, audioContext.sampleRate);
 *   if (freq) console.log(frequencyToNote(freq));
 */

/**
 * Estimate the fundamental frequency of a single time-domain buffer.
 *
 * @param {Float32Array} buffer - time-domain samples (e.g. from AnalyserNode.getFloatTimeDomainData)
 * @param {number} sampleRate - e.g. audioContext.sampleRate
 * @param {number} threshold - YIN absolute threshold (0.1-0.2 is typical; lower = stricter)
 * @returns {number|null} detected frequency in Hz, or null if no confident pitch found
 */
export function yinPitchDetect(buffer, sampleRate, threshold = 0.15) {
  const bufferSize = buffer.length;
  const halfBufferSize = Math.floor(bufferSize / 2);
  const yinBuffer = new Float32Array(halfBufferSize);

  // --- Step 1: difference function ---
  // For each lag tau, sum the squared difference between the signal
  // and itself shifted by tau. A true periodic signal will have a
  // sharp dip near tau = one period.
  for (let tau = 1; tau < halfBufferSize; tau++) {
    let sum = 0;
    for (let i = 0; i < halfBufferSize; i++) {
      const delta = buffer[i] - buffer[i + tau];
      sum += delta * delta;
    }
    yinBuffer[tau] = sum;
  }

  // --- Step 2: cumulative mean normalized difference ---
  // Normalizes the difference function so the threshold in step 3
  // is scale-invariant (works at any input volume).
  yinBuffer[0] = 1;
  let runningSum = 0;
  for (let tau = 1; tau < halfBufferSize; tau++) {
    runningSum += yinBuffer[tau];
    yinBuffer[tau] *= tau / runningSum;
  }

  // --- Step 3: absolute threshold ---
  // Walk forward until the normalized difference dips below the
  // threshold, then keep walking while it keeps improving (find the
  // local minimum, not just the first crossing).
  let tauEstimate = -1;
  for (let tau = 2; tau < halfBufferSize; tau++) {
    if (yinBuffer[tau] < threshold) {
      while (tau + 1 < halfBufferSize && yinBuffer[tau + 1] < yinBuffer[tau]) {
        tau++;
      }
      tauEstimate = tau;
      break;
    }
  }

  if (tauEstimate === -1) return null; // no periodic signal found (silence / noise)

  // --- Step 4: parabolic interpolation ---
  // Refines the integer-sample tau estimate to sub-sample precision
  // by fitting a parabola through the minimum and its neighbors.
  // This is what gets you accurate cents (not just note name).
  const x0 = tauEstimate < 1 ? tauEstimate : tauEstimate - 1;
  const x2 = tauEstimate + 1 < halfBufferSize ? tauEstimate + 1 : tauEstimate;

  let betterTau;
  if (x0 === tauEstimate) {
    betterTau = yinBuffer[tauEstimate] <= yinBuffer[x2] ? tauEstimate : x2;
  } else if (x2 === tauEstimate) {
    betterTau = yinBuffer[tauEstimate] <= yinBuffer[x0] ? tauEstimate : x0;
  } else {
    const s0 = yinBuffer[x0];
    const s1 = yinBuffer[tauEstimate];
    const s2 = yinBuffer[x2];
    betterTau = tauEstimate + (s2 - s0) / (2 * (2 * s1 - s2 - s0));
  }

  if (betterTau <= 0) return null;
  return sampleRate / betterTau;
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/**
 * Convert a frequency in Hz to the nearest musical note, octave, and
 * how far off (in cents) the detected pitch is from that note.
 *
 * @param {number} frequency - Hz
 * @returns {{noteName: string, octave: number, cents: number, frequency: number}}
 */
export function frequencyToNote(frequency) {
  // MIDI note number formula, A4 (440Hz) = MIDI note 69
  const midiFloat = 12 * Math.log2(frequency / 440) + 69;
  const midiRounded = Math.round(midiFloat);
  const cents = Math.round((midiFloat - midiRounded) * 100);

  const noteName = NOTE_NAMES[((midiRounded % 12) + 12) % 12];
  const octave = Math.floor(midiRounded / 12) - 1;

  return { noteName, octave, cents, frequency };
}

/**
 * Guitar standard tuning reference notes, useful for the tuner UI
 * and later as ground truth for the benchmark harness (step 4 of
 * the roadmap).
 */
export const STANDARD_TUNING = [
  { string: 6, note: 'E', octave: 2, frequency: 82.41 },
  { string: 5, note: 'A', octave: 2, frequency: 110.0 },
  { string: 4, note: 'D', octave: 3, frequency: 146.83 },
  { string: 3, note: 'G', octave: 3, frequency: 196.0 },
  { string: 2, note: 'B', octave: 3, frequency: 246.94 },
  { string: 1, note: 'E', octave: 4, frequency: 329.63 },
];
