import { Button } from './controls.tsx';

interface ArchitectureControlsProps {
  hiddenUnits: number[];
  onChange: (hiddenUnits: number[]) => void;
  maxLayers?: number;
  maxUnits?: number;
  minUnits?: number;
  disabled?: boolean;
}

/**
 * Add or remove hidden layers and set the width of each one. Used by every
 * section that lets the reader reshape the network.
 */
export function ArchitectureControls({
  hiddenUnits,
  onChange,
  maxLayers = 4,
  maxUnits = 12,
  minUnits = 1,
  disabled,
}: ArchitectureControlsProps) {
  const setUnits = (index: number, value: number) => {
    const next = hiddenUnits.slice();
    next[index] = value;
    onChange(next);
  };

  return (
    <div className="stack stack--sm">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <span className="control__label" style={{ margin: 0 }}>
          Hidden layers
          <span className="control__value" style={{ marginLeft: 8 }}>
            {hiddenUnits.length}
          </span>
        </span>
        <div className="btn-row">
          <Button
            onClick={() => onChange(hiddenUnits.slice(0, -1))}
            disabled={disabled || hiddenUnits.length === 0}
            title="Remove the last hidden layer"
          >
            − layer
          </Button>
          <Button
            onClick={() => onChange([...hiddenUnits, hiddenUnits[hiddenUnits.length - 1] ?? 4])}
            disabled={disabled || hiddenUnits.length >= maxLayers}
            title="Add a hidden layer"
          >
            + layer
          </Button>
        </div>
      </div>

      {hiddenUnits.length === 0 ? (
        <p className="faint" style={{ fontSize: 13, margin: 0 }}>
          No hidden layers: the network is a single linear map followed by the output activation.
        </p>
      ) : null}

      {hiddenUnits.map((units, index) => (
        <div className="control" key={index}>
          <label className="control__label">
            <span>Layer {index + 1} width</span>
            <span className="control__value">{units}</span>
          </label>
          <input
            type="range"
            min={minUnits}
            max={maxUnits}
            step={1}
            value={units}
            disabled={disabled}
            onChange={(event) => setUnits(index, Number(event.target.value))}
            aria-label={`Units in hidden layer ${index + 1}`}
          />
        </div>
      ))}
    </div>
  );
}
