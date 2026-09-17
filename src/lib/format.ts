/** Number formatting shared by every readout on the site. */

/** Fixed-decimal formatting that never prints "-0.00". */
export function fmt(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : value > 0 ? '∞' : '−∞';
  const rounded = Number(value.toFixed(digits));
  const text = (Object.is(rounded, -0) ? 0 : rounded).toFixed(digits);
  return text.replace('-', '−');
}

/** Same as fmt but with an explicit leading + for positive values. */
export function fmtSigned(value: number, digits = 2): string {
  const text = fmt(value, digits);
  return text.startsWith('−') ? text : `+${text}`;
}

/** Switches to scientific notation for very small or very large magnitudes. */
export function fmtAuto(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return fmt(value, digits);
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) {
    return value.toExponential(2).replace('-', '−').replace('e', 'e');
  }
  return fmt(value, digits);
}

export function fmtPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Subscript digits for indices such as x₁, w₂₃. */
const SUBSCRIPTS = '₀₁₂₃₄₅₆₇₈₉';

export function sub(n: number): string {
  return String(n)
    .split('')
    .map((c) => SUBSCRIPTS[Number(c)] ?? c)
    .join('');
}

/** Clamp helper used by every slider-driven computation. */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
