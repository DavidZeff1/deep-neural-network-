/**
 * Two-dimensional binary-classification datasets.
 *
 * All datasets live in the square [-1, 1] × [-1, 1] so that every plot and
 * decision-boundary sweep on the site shares one coordinate system.
 */

import type { Sample } from './network.ts';
import { makeGaussian, makeRng, shuffleInPlace } from './rng.ts';

export type DatasetName = 'linear' | 'xor' | 'circles' | 'moons' | 'spiral';

export interface DatasetOptions {
  count: number;
  /** Standard deviation of the Gaussian noise added to each point. */
  noise: number;
  seed: number;
}

export interface DatasetMeta {
  name: DatasetName;
  label: string;
  description: string;
  /** Smallest architecture that can separate it, stated plainly. */
  requirement: string;
}

export const DATASETS: Record<DatasetName, DatasetMeta> = {
  linear: {
    name: 'linear',
    label: 'Linear',
    description: 'Two clusters separated by a straight line.',
    requirement: 'Separable by a single neuron: one weight vector and a bias define the line.',
  },
  xor: {
    name: 'xor',
    label: 'XOR',
    description: 'Class 1 in the top-right and bottom-left quadrants, class 0 in the others.',
    requirement:
      'Not linearly separable. Needs at least one hidden layer with 2 units: two lines combined by the output neuron.',
  },
  circles: {
    name: 'circles',
    label: 'Circles',
    description: 'An inner disc surrounded by an outer ring.',
    requirement:
      'Needs a hidden layer. Roughly 3–4 ReLU units suffice to enclose the disc with a polygon.',
  },
  moons: {
    name: 'moons',
    label: 'Moons',
    description: 'Two interleaving crescents.',
    requirement: 'Needs a curved boundary; 4–8 hidden units produce a good fit.',
  },
  spiral: {
    name: 'spiral',
    label: 'Spiral',
    description: 'Two intertwined spiral arms.',
    requirement:
      'Requires high curvature. A single wide layer struggles; two hidden layers fit it with far fewer parameters.',
  },
};

export const DATASET_NAMES: DatasetName[] = ['linear', 'xor', 'circles', 'moons', 'spiral'];

export function generateDataset(name: DatasetName, options: DatasetOptions): Sample[] {
  const { count, noise, seed } = options;
  const rng = makeRng(seed);
  const gaussian = makeGaussian(rng);
  const samples: Sample[] = [];
  const jitter = (v: number) => v + gaussian() * noise;

  switch (name) {
    case 'linear': {
      for (let i = 0; i < count; i++) {
        const label = i % 2;
        // Two clusters offset along the (1, 1)/√2 direction.
        const offset = label === 1 ? 0.45 : -0.45;
        const along = (rng() * 2 - 1) * 0.75;
        const x1 = offset * 0.85 + along * 0.7;
        const x2 = offset * 0.85 - along * 0.7;
        samples.push({ x: [jitter(x1), jitter(x2)], y: [label] });
      }
      break;
    }
    case 'xor': {
      for (let i = 0; i < count; i++) {
        const sx = i % 2 === 0 ? 1 : -1;
        const sy = i % 4 < 2 ? 1 : -1;
        const x1 = sx * (0.12 + rng() * 0.78);
        const x2 = sy * (0.12 + rng() * 0.78);
        const label = sx * sy > 0 ? 1 : 0;
        samples.push({ x: [jitter(x1), jitter(x2)], y: [label] });
      }
      break;
    }
    case 'circles': {
      for (let i = 0; i < count; i++) {
        const label = i % 2;
        const radius = label === 1 ? rng() * 0.36 : 0.62 + rng() * 0.28;
        const angle = rng() * Math.PI * 2;
        samples.push({
          x: [jitter(radius * Math.cos(angle)), jitter(radius * Math.sin(angle))],
          y: [label],
        });
      }
      break;
    }
    case 'moons': {
      for (let i = 0; i < count; i++) {
        const label = i % 2;
        const angle = rng() * Math.PI;
        if (label === 1) {
          samples.push({
            x: [jitter(Math.cos(angle) * 0.8 - 0.35), jitter(Math.sin(angle) * 0.8 - 0.25)],
            y: [1],
          });
        } else {
          samples.push({
            x: [jitter(Math.cos(angle) * 0.8 + 0.35), jitter(-Math.sin(angle) * 0.8 + 0.25)],
            y: [0],
          });
        }
      }
      break;
    }
    case 'spiral': {
      for (let i = 0; i < count; i++) {
        const label = i % 2;
        const t = (i / count) * 2.6 + 0.35;
        const angle = t * 2.2 + label * Math.PI;
        const radius = t * 0.34;
        samples.push({
          x: [jitter(radius * Math.cos(angle)), jitter(radius * Math.sin(angle))],
          y: [label],
        });
      }
      break;
    }
  }

  return shuffleInPlace(samples, rng);
}

export interface SplitDataset {
  train: Sample[];
  validation: Sample[];
}

/** Deterministic split; `trainFraction` of the points go to the training set. */
export function splitDataset(samples: Sample[], trainFraction: number, seed = 7): SplitDataset {
  const rng = makeRng(seed);
  const shuffled = shuffleInPlace(samples.slice(), rng);
  const cut = Math.max(1, Math.round(shuffled.length * trainFraction));
  return { train: shuffled.slice(0, cut), validation: shuffled.slice(cut) };
}
