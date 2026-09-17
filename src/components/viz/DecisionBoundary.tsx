import { useEffect, useRef, useState } from 'react';
import type { Sample } from '../../lib/network.ts';
import { marchingSquares } from '../../lib/contour.ts';
import { useThemeSignal } from '../../hooks/useThemeSignal.ts';

interface DecisionBoundaryProps {
  /** Returns the network's output probability for a point in data space. */
  predict: (x1: number, x2: number) => number;
  train: Sample[];
  validation?: Sample[];
  /** Change this whenever `predict` produces different values. */
  version: number;
  domain?: [number, number];
  resolution?: number;
  showPoints?: boolean;
  hint?: string;
  onPick?: (x1: number, x2: number) => void;
  highlight?: [number, number] | null;
}

const CLASS_A = [59, 130, 246] as const; // class 1 — blue
const CLASS_B = [224, 118, 31] as const; // class 0 — orange

function readRgb(el: HTMLElement, variable: string, fallback: [number, number, number]) {
  const raw = getComputedStyle(el).getPropertyValue(variable).trim();
  const match = /^#([0-9a-f]{6})$/i.exec(raw);
  if (!match) return fallback;
  const value = parseInt(match[1], 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255] as [number, number, number];
}

export function DecisionBoundary({
  predict,
  train,
  validation,
  version,
  domain = [-1.25, 1.25],
  resolution = 64,
  showPoints = true,
  hint,
  onPick,
  highlight,
}: DecisionBoundaryProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState(360);
  const themeVersion = useThemeSignal();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? 0;
      if (width > 0) setSize(Math.round(width));
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const pixels = Math.max(160, Math.round(size * dpr));
    if (canvas.width !== pixels) {
      canvas.width = pixels;
      canvas.height = pixels;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bg = readRgb(canvas, '--bg-inset', [242, 244, 247]);
    const ink = readRgb(canvas, '--text', [20, 23, 31]);
    const isDark = document.documentElement.dataset.theme === 'dark';
    // A dark background needs less tint to reach the same perceived contrast.
    const tint = isDark ? 0.34 : 0.5;
    const [lo, hi] = domain;
    const span = hi - lo;
    const toPixel = (v: number) => ((v - lo) / span) * pixels;

    // 1. Probability field.
    const grid = new Float64Array(resolution * resolution);
    for (let r = 0; r < resolution; r++) {
      const x2 = hi - (span * r) / (resolution - 1);
      for (let c = 0; c < resolution; c++) {
        const x1 = lo + (span * c) / (resolution - 1);
        grid[r * resolution + c] = predict(x1, x2);
      }
    }

    const field = ctx.createImageData(resolution, resolution);
    for (let i = 0; i < grid.length; i++) {
      const p = Math.min(1, Math.max(0, grid[i]));
      const target = p >= 0.5 ? CLASS_A : CLASS_B;
      const alpha = Math.min(1, Math.abs(p - 0.5) * 2) * tint;
      field.data[i * 4] = Math.round(bg[0] * (1 - alpha) + target[0] * alpha);
      field.data[i * 4 + 1] = Math.round(bg[1] * (1 - alpha) + target[1] * alpha);
      field.data[i * 4 + 2] = Math.round(bg[2] * (1 - alpha) + target[2] * alpha);
      field.data[i * 4 + 3] = 255;
    }

    // Upscale the low-resolution field smoothly.
    const offscreen = document.createElement('canvas');
    offscreen.width = resolution;
    offscreen.height = resolution;
    offscreen.getContext('2d')?.putImageData(field, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, pixels, pixels);
    ctx.drawImage(offscreen, 0, 0, pixels, pixels);

    // 2. The p = 0.5 contour.
    const segments = marchingSquares(grid, resolution, resolution, 0.5);
    const cell = pixels / (resolution - 1);
    ctx.lineWidth = Math.max(1.2, dpr * 1.1);
    ctx.strokeStyle = `rgba(${ink[0]}, ${ink[1]}, ${ink[2]}, 0.55)`;
    ctx.beginPath();
    for (const [x1, y1, x2, y2] of segments) {
      ctx.moveTo(x1 * cell, y1 * cell);
      ctx.lineTo(x2 * cell, y2 * cell);
    }
    ctx.stroke();

    // 3. Data points. Validation points are drawn hollow.
    if (showPoints) {
      const radius = Math.max(2.4, pixels / 110);
      const drawSet = (samples: Sample[], filled: boolean) => {
        for (const sample of samples) {
          const px = toPixel(sample.x[0]);
          const py = pixels - toPixel(sample.x[1]);
          const colour = sample.y[0] >= 0.5 ? CLASS_A : CLASS_B;
          ctx.beginPath();
          ctx.arc(px, py, radius, 0, Math.PI * 2);
          if (filled) {
            ctx.fillStyle = `rgb(${colour[0]}, ${colour[1]}, ${colour[2]})`;
            ctx.fill();
            ctx.lineWidth = Math.max(0.8, dpr * 0.7);
            ctx.strokeStyle = isDark ? 'rgba(14, 17, 22, 0.8)' : 'rgba(255, 255, 255, 0.85)';
            ctx.stroke();
          } else {
            ctx.lineWidth = Math.max(1.4, dpr * 1.2);
            ctx.strokeStyle = `rgb(${colour[0]}, ${colour[1]}, ${colour[2]})`;
            ctx.stroke();
          }
        }
      };
      drawSet(train, true);
      if (validation) drawSet(validation, false);
    }

    if (highlight) {
      const px = toPixel(highlight[0]);
      const py = pixels - toPixel(highlight[1]);
      ctx.beginPath();
      ctx.arc(px, py, Math.max(5, pixels / 60), 0, Math.PI * 2);
      ctx.lineWidth = Math.max(1.6, dpr * 1.4);
      ctx.strokeStyle = `rgb(${ink[0]}, ${ink[1]}, ${ink[2]})`;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px, py, Math.max(2, pixels / 150), 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${ink[0]}, ${ink[1]}, ${ink[2]})`;
      ctx.fill();
    }
  }, [predict, train, validation, version, size, domain, resolution, showPoints, highlight, themeVersion]);

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        className="boundary-canvas"
        style={onPick ? { cursor: 'crosshair' } : undefined}
        onPointerDown={(event) => {
          if (!onPick) return;
          const rect = event.currentTarget.getBoundingClientRect();
          const [lo, hi] = domain;
          const span = hi - lo;
          const x1 = lo + ((event.clientX - rect.left) / rect.width) * span;
          const x2 = hi - ((event.clientY - rect.top) / rect.height) * span;
          onPick(x1, x2);
        }}
      />
      {hint ? <span className="canvas-wrap__hint">{hint}</span> : null}
    </div>
  );
}
