import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig } from '../lib/network.ts';
import { DATASETS, DATASET_NAMES, generateDataset } from '../lib/datasets.ts';
import type { DatasetName } from '../lib/datasets.ts';
import { ACTIVATIONS, HIDDEN_ACTIVATIONS } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { useEnsembleTrainer } from '../hooks/useEnsembleTrainer.ts';
import type { EnsembleMember } from '../hooks/useEnsembleTrainer.ts';
import { DecisionBoundary } from '../components/viz/DecisionBoundary.tsx';
import { MetricChart } from '../components/viz/MetricChart.tsx';
import { Panel, Note, Stats } from '../components/ui/layout.tsx';
import { Button, SelectField, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { fmt, fmtPercent } from '../lib/format.ts';

const SERIES_COLORS = ['#94a3b8', '#e0761f', '#3b82f6', '#16a34a'];

/** Exact upper bound on the linear regions of a one-hidden-layer ReLU net in d inputs. */
function shallowRegions(units: number, inputs = 2): number {
  let total = 0;
  for (let j = 0; j <= inputs; j++) {
    let binomial = 1;
    for (let k = 0; k < j; k++) binomial = (binomial * (units - k)) / (k + 1);
    total += binomial;
  }
  return Math.round(total);
}

/** Lower bound for an L-hidden-layer ReLU net of width n on d inputs. */
function deepRegions(units: number, layers: number, inputs = 2): number {
  const base = Math.floor(units / inputs) ** inputs;
  return Math.round(base ** (layers - 1) * shallowRegions(units, inputs));
}

export function DepthSection({ id, index }: SectionProps) {
  const [datasetName, setDatasetName] = useState<DatasetName>('spiral');
  const [width, setWidth] = useState(6);
  const [activation, setActivation] = useState<ActivationName>('relu');
  const [learningRate, setLearningRate] = useState(0.25);
  const [selected, setSelected] = useState(2);
  const [counterWidth, setCounterWidth] = useState(8);

  const train = useMemo(
    () => generateDataset(datasetName, { count: 240, noise: 0.05, seed: 23 }),
    [datasetName],
  );

  const members = useMemo<EnsembleMember[]>(() => {
    const base: Omit<NetworkConfig, 'hiddenUnits'> = {
      inputSize: 2,
      outputSize: 1,
      hiddenActivation: activation,
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 808,
    };
    return [
      { label: 'Input → Output', config: { ...base, hiddenUnits: [] } },
      { label: 'Input → H → Output', config: { ...base, hiddenUnits: [width] } },
      { label: 'Input → H → H → Output', config: { ...base, hiddenUnits: [width, width] } },
      { label: 'Input → H → H → H → Output', config: { ...base, hiddenUnits: [width, width, width] } },
    ];
  }, [activation, width]);

  const trainer = useEnsembleTrainer(members, train, learningRate, 16, 2, 1500);

  const lossSeries = useMemo(
    () =>
      trainer.results.map((result, i) => ({
        label: result.label,
        color: SERIES_COLORS[i],
        values: result.history,
      })),
    [trainer.results],
  );

  const selectedResult = trainer.results[selected];
  const hiddenLayerIndex = selectedResult.network.sizes.length - 2;
  const hiddenCount = hiddenLayerIndex >= 1 ? selectedResult.network.sizes[hiddenLayerIndex] : 0;

  return (
    <Section
      id={id}
      index={index}
      title="Why depth matters"
      lede={
        <>
          Four networks, identical except for the number of hidden layers, trained on the same data
          with the same learning rate and the same number of epochs. The difference in what they can
          represent is visible in the boundaries they produce.
        </>
      }
    >
      <div className="grid grid--side">
        <Panel
          title="Same data, increasing depth"
          hint={`epoch ${trainer.epoch}`}
          caption="All four start from the same random seed and receive the same updates per epoch. Only the number of hidden layers differs."
        >
          <div className="grid grid--4" style={{ gap: 12 }}>
            {trainer.results.map((result, i) => (
              <button
                key={result.label}
                type="button"
                onClick={() => setSelected(i)}
                style={{
                  border: i === selected ? '1px solid var(--accent)' : '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  background: 'transparent',
                  padding: 6,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                <DecisionBoundary
                  predict={(x1, x2) => result.network.forward([x1, x2]).output[0]}
                  train={train}
                  version={trainer.version}
                  resolution={48}
                />
                <div style={{ marginTop: 6 }}>
                  <div className="mono" style={{ fontSize: 11, color: 'var(--text)' }}>
                    {result.network.sizes.join(' → ')}
                  </div>
                  <div className="mono" style={{ fontSize: 11, color: 'var(--text-faint)' }}>
                    loss {fmt(result.loss, 3)} · {fmtPercent(result.accuracy, 0)} ·{' '}
                    {result.parameterCount}p
                  </div>
                </div>
              </button>
            ))}
          </div>
        </Panel>

        <div className="stack">
          <Panel title="Run">
            <div className="btn-row">
              <Button variant="primary" onClick={() => (trainer.running ? trainer.pause() : trainer.start())}>
                {trainer.running ? 'Pause' : 'Train all four'}
              </Button>
              <Button onClick={() => trainer.runEpochs(25)}>+25 epochs</Button>
              <Button onClick={trainer.reset}>Reset</Button>
            </div>
            <div className="stack stack--sm" style={{ marginTop: 14 }}>
              <SelectField
                label="Dataset"
                value={datasetName}
                onChange={setDatasetName}
                options={DATASET_NAMES.map((name) => ({ value: name, label: DATASETS[name].label }))}
              />
              <Slider label="Units per hidden layer" min={2} max={12} step={1} value={width} onChange={setWidth} display={width} />
              <Slider
                label="Learning rate η"
                min={0.01}
                max={1}
                step={0.01}
                value={learningRate}
                onChange={setLearningRate}
                display={fmt(learningRate, 2)}
              />
              <Segmented
                label="Activation"
                value={activation}
                options={HIDDEN_ACTIVATIONS.map((n) => ({ value: n, label: ACTIVATIONS[n].label }))}
                onChange={setActivation}
              />
            </div>
          </Panel>

          <Panel title="Training loss, all four" flush>
            <div style={{ padding: '10px 10px 0' }}>
              <MetricChart series={lossSeries} yMin={0} yLabel="loss" height={200} ariaLabel="Loss by depth" />
            </div>
            <div className="figure__caption">
              {trainer.results.map((result, i) => (
                <span key={result.label} style={{ marginRight: 14, whiteSpace: 'nowrap' }}>
                  <span
                    className="legend__swatch"
                    style={{ background: SERIES_COLORS[i], display: 'inline-block', marginRight: 5 }}
                  />
                  {result.network.sizes.length - 1} layer{result.network.sizes.length - 1 === 1 ? '' : 's'}
                </span>
              ))}
            </div>
          </Panel>
        </div>
      </div>

      <div className="grid grid--side">
        <Panel
          title={`Hidden unit responses — ${selectedResult.label}`}
          hint="last hidden layer"
          caption="Each tile shows one unit's activation across the input plane; the dark curve marks where that unit's output is zero. These are the features the next layer combines. Click a boundary above to switch network."
        >
          {hiddenCount === 0 ? (
            <p className="faint" style={{ fontSize: 13, margin: 0 }}>
              This network has no hidden layer. Its output is a single sigmoid applied to a linear
              function of the inputs, so there are no intermediate features to show.
            </p>
          ) : (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${Math.min(6, hiddenCount)}, minmax(0, 1fr))`,
                gap: 8,
              }}
            >
              {Array.from({ length: hiddenCount }, (_, unit) => (
                <div key={unit}>
                  <DecisionBoundary
                    predict={(x1, x2) => {
                      const trace = selectedResult.network.forward([x1, x2]);
                      const a = trace.layers[hiddenLayerIndex - 1].a[unit];
                      return 1 / (1 + Math.exp(-2 * a));
                    }}
                    train={[]}
                    version={trainer.version}
                    resolution={36}
                    showPoints={false}
                  />
                  <div className="mono faint" style={{ fontSize: 10, textAlign: 'center', marginTop: 2 }}>
                    a{unit + 1}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>

        <div className="prose-block">
          <h3 className="subhead">Depth as composition</h3>
          <p>
            A network is a composition <M>{'f_L \\circ \\cdots \\circ f_1'}</M>. Each{' '}
            <M>{'f_l'}</M> is an affine map followed by a non-linearity. With ReLU, every{' '}
            <M>{'f_l'}</M> is piecewise linear, so the whole network is piecewise linear: the input
            space is partitioned into regions, and on each region the network is exactly an affine
            function.
          </p>
          <p>
            A single hidden layer with <M>{'n'}</M> units places <M>{'n'}</M> hyperplanes in the
            input space. In <M>{'d = 2'}</M> dimensions they cut the plane into at most
          </p>
          <Equation plain>{'\\binom{n}{0} + \\binom{n}{1} + \\binom{n}{2}'}</Equation>
          <p>
            regions — quadratic in <M>{'n'}</M>. Adding a second layer does not add hyperplanes; it
            folds the regions produced by the first layer onto each other, so each subsequent layer
            multiplies the region count rather than adding to it. The growth becomes exponential in
            depth and polynomial in width.
          </p>
        </div>
      </div>

      <div className="grid grid--side">
        <Panel title="Region count, d = 2 inputs" flush>
          <div style={{ padding: '14px 16px 4px' }}>
            <Slider
              label="Width n"
              min={2}
              max={20}
              step={1}
              value={counterWidth}
              onChange={setCounterWidth}
              display={counterWidth}
            />
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>Hidden layers</th>
                <th>Parameters</th>
                <th>Linear regions</th>
              </tr>
            </thead>
            <tbody>
              {[1, 2, 3, 4].map((layers) => {
                const n = counterWidth;
                const params =
                  2 * n + n + (layers - 1) * (n * n + n) + n + 1;
                const regions = layers === 1 ? shallowRegions(n) : deepRegions(n, layers);
                return (
                  <tr key={layers}>
                    <td>{layers}</td>
                    <td>{params}</td>
                    <td>{regions.toLocaleString('en-US')}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="figure__caption">
            Row 1 is the exact maximum for one hidden layer. Rows 2–4 are lower bounds from the
            standard construction: each additional layer contributes a factor of ⌊n/d⌋ᵈ. Parameter
            count grows linearly in depth while the region count grows exponentially.
          </div>
        </Panel>

        <div className="prose-block">
          <h3 className="subhead">What depth does not give you</h3>
          <p>
            The universal approximation theorem states that one hidden layer with enough units can
            approximate any continuous function on a compact set to arbitrary accuracy. Depth is
            therefore not about what is representable in principle — it is about how many units are
            required.
          </p>
          <p>
            For some function families the gap is exponential: a function computable by a deep
            network of width <M>{'n'}</M> and depth <M>{'L'}</M> can require width exponential in{' '}
            <M>{'L'}</M> to compute with a single hidden layer. Depth buys parameter efficiency, not
            new expressive limits.
          </p>
          <p>
            Depth also costs something. Each additional layer adds a factor of{' '}
            <M>{"f'(z)"}</M> and <M>{'W^{\\top}'}</M> to the backward product, so gradients decay or
            explode more easily. Set the activation above to sigmoid and train: the three-layer
            network now learns more slowly than the one-layer network, despite being strictly more
            expressive.
          </p>
        </div>
      </div>

      <Stats
        items={trainer.results.map((result) => ({
          label: `${result.network.sizes.length - 1}L loss`,
          value: fmt(result.loss, 3),
        }))}
      />

      <Note title="Reading the comparison honestly" accent>
        <p>
          At equal epochs the deeper networks usually reach a lower loss on Spiral and Circles and
          make no difference on Linear, where a single boundary is already optimal. The comparison
          holds only for this training budget: given enough units and epochs the shallow network can
          also fit Spiral. The claim depth supports is about efficiency — fewer parameters and fewer
          epochs for the same fit — not about possibility.
        </p>
      </Note>
    </Section>
  );
}
