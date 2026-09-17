import type { ReactNode } from 'react';
import { useId } from 'react';

interface SliderProps {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  /** Rendered next to the label; defaults to the raw value. */
  display?: ReactNode;
  disabled?: boolean;
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  display,
  disabled,
}: SliderProps) {
  const id = useId();
  return (
    <div className="control">
      <label className="control__label" htmlFor={id}>
        <span>{label}</span>
        <span className="control__value">{display ?? value}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

interface Option<T extends string> {
  value: T;
  label: string;
}

interface SelectFieldProps<T extends string> {
  label: ReactNode;
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
}

export function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled,
}: SelectFieldProps<T>) {
  const id = useId();
  return (
    <div className="control">
      <label className="control__label" htmlFor={id}>
        <span>{label}</span>
      </label>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value as T)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface SegmentedProps<T extends string> {
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
  label?: ReactNode;
  ariaLabel?: string;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  ariaLabel,
}: SegmentedProps<T>) {
  const control = (
    <div className="segmented" role="radiogroup" aria-label={ariaLabel ?? (typeof label === 'string' ? label : undefined)}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={option.value === value}
          className={
            option.value === value ? 'segmented__option segmented__option--active' : 'segmented__option'
          }
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
  if (!label) return control;
  return (
    <div className="control">
      <span className="control__label">{label}</span>
      {control}
    </div>
  );
}

interface ButtonProps {
  children: ReactNode;
  onClick: () => void;
  variant?: 'default' | 'primary';
  disabled?: boolean;
  title?: string;
}

export function Button({ children, onClick, variant = 'default', disabled, title }: ButtonProps) {
  return (
    <button
      type="button"
      title={title}
      className={variant === 'primary' ? 'btn btn--primary' : 'btn'}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

interface CheckboxProps {
  label: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

export function Checkbox({ label, checked, onChange, disabled }: CheckboxProps) {
  return (
    <label className="checkbox">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

interface NumberFieldProps {
  label: ReactNode;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export function NumberField({ label, value, onChange, min, max, step = 0.1 }: NumberFieldProps) {
  const id = useId();
  return (
    <div className="control">
      <label className="control__label" htmlFor={id}>
        <span>{label}</span>
      </label>
      <input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) onChange(next);
        }}
      />
    </div>
  );
}
