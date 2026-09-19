import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { ForwardTrace, MLP, NetworkConfig, Sample } from '../lib/network.ts';
import { DATASETS, DATASET_NAMES, generateDataset, splitDataset } from '../lib/datasets.ts';
import type { DatasetName } from '../lib/datasets.ts';
import { ACTIVATIONS, HIDDEN_ACTIVATIONS } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { useTrainer } from '../hooks/useTrainer.ts';
import { NetworkDiagram } from '../components/viz/NetworkDiagram.tsx';
import type { DiagramSelection } from '../components/viz/NetworkDiagram.tsx';
import { DecisionBoundary } from '../components/viz/DecisionBoundary.tsx';
import { MetricChart } from '../components/viz/MetricChart.tsx';
import { ArchitectureControls } from '../components/ui/ArchitectureControls.tsx';
import { Panel, Note, Stats, Legend } from '../components/ui/layout.tsx';
import { Detail } from '../components/ui/Detail.tsx';
import { Button, Checkbox, SelectField, Segmented, Slider } from '../components/ui/controls.tsx';
import { M } from '../components/ui/Math.tsx';
import { Exercise } from '../components/ui/Exercise.tsx';
import { fmt, fmtPercent } from '../lib/format.ts';

const TRAIN_COLOR = '#3b82f6';
const VAL_COLOR = '#e0761f';

/** Mean ∂L/∂W over a dataset, without applying any update. */
function meanGradient(network: MLP, samples: Sample[]): number[][][] {
  const accumulator = network.W.map((layer) => layer.map((row) => row.map(() => 0)));
  if (samples.length === 0) return accumulator;
  for (const sample of samples) {
    const grads = network.backward(network.forward(sample.x), sample.y);
    for (let l = 0; l < accumulator.length; l++) {
      for (let j = 0; j < accumulator[l].length; j++) {
        for (let i = 0; i < accumulator[l][j].length; i++) {
          accumulator[l][j][i] += grads.dW[l][j][i] / samples.length;
        }
      }
    }
  }
  return accumulator;
}

interface InspectorProps {
  selection: DiagramSelection | null;
  network: MLP;
  trace: ForwardTrace;
  probe: [number, number];
  gradients: number[][][] | null;
  activation: ActivationName;
  learningRate: number;
}

/** Shows the arithmetic behind whichever unit or connection is selected. */
function Inspector({
  selection,
  network,
  trace,
  probe,
  gradients,
  activation,
  learningRate,
}: InspectorProps) {
  if (selection?.kind === 'node') {
    const { layer, index: unit } = selection;
    if (layer === 0) {
      return (
        <div className="calc">
          <div className="calc__line">
            <span className="calc__label">input x{unit + 1} = </span>
            <span className="calc__result">{fmt(probe[unit], 4)}</span>
          </div>
          <div className="calc__line calc__label">
            Inputs are not computed; they are the coordinates of the probe point.
          </div>
        </div>
      );
    }
    const z = trace.layers[layer - 1].z[unit];
    const a = trace.layers[layer - 1].a[unit];
    const weights = network.W[layer - 1][unit];
    const bias = network.b[layer - 1][unit];
    const inputs = layer === 1 ? trace.input : trace.layers[layer - 2].a;
    const isOutput = layer === network.sizes.length - 1;
    const fname = isOutput ? 'σ' : ACTIVATIONS[activation].label;
    return (
      <div className="calc">
        <div className="calc__line">
          <span className="calc__label">inputs   </span>[{inputs.map((v) => fmt(v, 3)).join(', ')}]
        </div>
        <div className="calc__line">
          <span className="calc__label">weights  </span>[{weights.map((v) => fmt(v, 3)).join(', ')}]
        </div>
        <div className="calc__line">
          <span className="calc__label">bias     </span>{fmt(bias, 3)}
        </div>
        <div className="calc__line">
          <span className="calc__label">z = </span>
          {weights.map((w, i) => `(${fmt(w, 2)})(${fmt(inputs[i], 2)})`).join(' + ')} + {fmt(bias, 2)}
        </div>
        <div className="calc__line">
          <span className="calc__label">z = </span>
          <span className="calc__result">{fmt(z, 4)}</span>
        </div>
        <div className="calc__line">
          <span className="calc__label">a = </span>
          {fname}({fmt(z, 4)}) = <span className="calc__result">{fmt(a, 4)}</span>
        </div>
        <div className="calc__line">
          <span className="calc__label">f'(z) = </span>
          {isOutput ? fmt(a * (1 - a), 4) : fmt(ACTIVATIONS[activation].df(z), 4)}
        </div>
      </div>
    );
  }

  if (selection?.kind === 'edge') {
    const { layer, from, to } = selection;
    const weight = network.W[layer][to][from];
    const gradient = gradients?.[layer]?.[to]?.[from] ?? 0;
    const source = layer === 0 ? trace.input[from] : trace.layers[layer - 1].a[from];
    return (
      <div className="calc">
        <div className="calc__line">
          <span className="calc__label">weight w  </span>
          <span className="calc__result">{fmt(weight, 4)}</span>
        </div>
        <div className="calc__line">
          <span className="calc__label">from      </span>layer {layer} unit {from + 1} (a ={' '}
          {fmt(source, 3)})
        </div>
        <div className="calc__line">
          <span className="calc__label">to        </span>layer {layer + 1} unit {to + 1}
        </div>
        <div className="calc__line">
          <span className="calc__label">contributes </span>({fmt(weight, 3)})({fmt(source, 3)}) ={' '}
          {fmt(weight * source, 4)} to z
        </div>
        <div className="calc__line">
          <span className="calc__label">mean ∂L/∂w </span>= <span className="calc__result">{fmt(gradient, 5)}</span>
        </div>
        <div className="calc__line">
          <span className="calc__label">w − η·∂L/∂w</span> = {fmt(weight - learningRate * gradient, 5)}
        </div>
      </div>
    );
  }

  return (
    <p className="faint" style={{ fontSize: 13, margin: 0 }}>
      Click a unit or a connection in the network diagram. Units show the weighted sum, the bias and
      the activation for the probe point; connections show the weight, its contribution to the next
      pre-activation, and its mean gradient over the training set.
    </p>
  );
}

export function PlaygroundSection({ id, index }: SectionProps) {
  const [datasetName, setDatasetName] = useState<DatasetName>('moons');
  const [count, setCount] = useState(240);
  const [noise, setNoise] = useState(0.14);
  const [useValidation, setUseValidation] = useState(true);
  const [hiddenUnits, setHiddenUnits] = useState<number[]>([6, 4]);
  const [activation, setActivation] = useState<ActivationName>('tanh');
  const [learningRate, setLearningRate] = useState(0.15);
  const [batchSize, setBatchSize] = useState(16);
  const [l2, setL2] = useState(0);
  const [dropout, setDropout] = useState(0);
  const [speed, setSpeed] = useState(3);
  const [probe, setProbe] = useState<[number, number]>([0.4, 0.3]);
  const [selection, setSelection] = useState<DiagramSelection | null>(null);

  const { train, validation } = useMemo(() => {
    const all = generateDataset(datasetName, { count, noise, seed: 55 });
    if (!useValidation) return { train: all, validation: [] as Sample[] };
    return splitDataset(all, 0.75, 9);
  }, [datasetName, count, noise, useValidation]);

  const config = useMemo<NetworkConfig>(
    () => ({
      inputSize: 2,
      hiddenUnits,
      outputSize: 1,
      hiddenActivation: activation,
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 2718,
      l2,
      dropout,
    }),
    [hiddenUnits, activation, l2, dropout],
  );

  const trainer = useTrainer({
    config,
    train,
    validation,
    learningRate,
    batchSize,
    epochsPerTick: speed,
    maxEpochs: 4000,
  });

  const trace = useMemo(
    () => trainer.network.forward(probe),
    // `version` is the dependency that matters: the network is mutated in place.
    [trainer.network, probe, trainer.version],
  );

  const gradients = useMemo(
    () => (selection?.kind === 'edge' ? meanGradient(trainer.network, train) : null),
    // `version` is the dependency that matters: the network is mutated in place.
    [selection, trainer.network, train, trainer.version],
  );

  const predict = useMemo(
    () => (x1: number, x2: number) => trainer.network.forward([x1, x2]).output[0],
    // `version` is the dependency that matters: the network is mutated in place.
    [trainer.network, trainer.version],
  );

  const lossSeries = useMemo(() => {
    const series = [
      {
        label: 'training',
        color: TRAIN_COLOR,
        values: trainer.history.map((p) => [p.epoch, p.trainLoss] as [number, number]),
      },
    ];
    if (validation.length > 0) {
      series.push({
        label: 'validation',
        color: VAL_COLOR,
        dash: '5 4',
        values: trainer.history.map((p) => [p.epoch, p.validationLoss] as [number, number]),
      } as (typeof series)[number]);
    }
    return series;
  }, [trainer.history, validation.length]);

  const accuracySeries = useMemo(() => {
    const series = [
      {
        label: 'training',
        color: TRAIN_COLOR,
        values: trainer.history.map((p) => [p.epoch, p.trainAccuracy] as [number, number]),
      },
    ];
    if (validation.length > 0) {
      series.push({
        label: 'validation',
        color: VAL_COLOR,
        dash: '5 4',
        values: trainer.history.map((p) => [p.epoch, p.validationAccuracy] as [number, number]),
      } as (typeof series)[number]);
    }
    return series;
  }, [trainer.history, validation.length]);

  const inspector = (
    <Inspector
      selection={selection}
      network={trainer.network}
      trace={trace}
      probe={probe}
      gradients={gradients}
      activation={activation}
      learningRate={learningRate}
    />
  );

  return (
    <Section
      id={id}
      index={index}
      title="Playground"
      lede={
        <>
          Everything from the previous sections in one place: configure a network and a dataset,
          train it, probe any point of the input plane, and inspect any unit or weight while the
          parameters change.
        </>
      }
    >
      <div className="grid grid--side">
        <div className="stack">
          <Panel
            title="Decision boundary and probe"
            hint={trainer.running ? 'training…' : `epoch ${trainer.metrics.epoch}`}
            caption="Click anywhere on the plane to move the probe. The network diagram below shows the forward pass for that exact point."
          >
            <div className="grid grid--canvas" style={{ gap: 16 }}>
              <DecisionBoundary
                predict={predict}
                train={train}
                validation={validation}
                version={trainer.version}
                resolution={72}
                highlight={probe}
                onPick={(x1, x2) => setProbe([Number(x1.toFixed(3)), Number(x2.toFixed(3))])}
                hint="click to probe"
              />
              <div className="stack stack--sm">
                <Stats
                  items={[
                    { label: 'Probe x₁', value: fmt(probe[0], 3) },
                    { label: 'Probe x₂', value: fmt(probe[1], 3) },
                    { label: 'ŷ at probe', value: fmt(trace.output[0], 4), accent: true },
                    { label: 'Predicted class', value: trace.output[0] >= 0.5 ? '1' : '0' },
                  ]}
                />
                <Slider
                  label="Probe x₁"
                  min={-1.25}
                  max={1.25}
                  step={0.01}
                  value={probe[0]}
                  onChange={(v) => setProbe([v, probe[1]])}
                  display={fmt(probe[0], 2)}
                />
                <Slider
                  label="Probe x₂"
                  min={-1.25}
                  max={1.25}
                  step={0.01}
                  value={probe[1]}
                  onChange={(v) => setProbe([probe[0], v])}
                  display={fmt(probe[1], 2)}
                />
                <Legend
                  items={[
                    { color: TRAIN_COLOR, label: 'class 1' },
                    { color: VAL_COLOR, label: 'class 0' },
                    { color: 'var(--text-faint)', label: 'hollow = validation', outline: true },
                  ]}
                />
              </div>
            </div>
          </Panel>

          <Panel
            title="Network"
            hint="click a unit or a connection"
            caption="Node shading is the activation at the probe point; edge thickness is |w|."
          >
            <NetworkDiagram
              network={trainer.network}
              trace={trace}
              selection={selection}
              onSelect={setSelection}
              height={260}
              inputLabels={['x₁', 'x₂']}
              outputLabels={['ŷ']}
            />
          </Panel>

          <div className="grid grid--2">
            <Panel title="Loss" flush>
              <div style={{ padding: '10px 10px 0' }}>
                <MetricChart series={lossSeries} yMin={0} yLabel="loss" ariaLabel="Loss curves" />
              </div>
            </Panel>
            <Panel title="Accuracy" flush>
              <div style={{ padding: '10px 10px 0' }}>
                <MetricChart series={accuracySeries} yMin={0} yMax={1.02} yLabel="accuracy" ariaLabel="Accuracy curves" />
              </div>
            </Panel>
          </div>
        </div>

        <div className="stack">
          <Panel title="Run">
            <div className="btn-row">
              <Button variant="primary" onClick={() => (trainer.running ? trainer.pause() : trainer.start())}>
                {trainer.running ? 'Pause' : 'Train'}
              </Button>
              <Button onClick={() => trainer.runEpochs(1)}>+1</Button>
              <Button onClick={() => trainer.runEpochs(25)}>+25</Button>
              <Button onClick={() => trainer.reset()}>Reset</Button>
              <Button onClick={() => trainer.reset(Math.floor(Math.random() * 100000))}>New init</Button>
            </div>
            <div style={{ marginTop: 14 }}>
              <Stats
                items={[
                  { label: 'Epoch', value: trainer.metrics.epoch },
                  { label: 'Train loss', value: fmt(trainer.metrics.trainLoss, 4), accent: true },
                  { label: 'Train acc', value: fmtPercent(trainer.metrics.trainAccuracy, 1) },
                  ...(validation.length > 0
                    ? [
                        { label: 'Val loss', value: fmt(trainer.metrics.validationLoss, 4) },
                        { label: 'Val acc', value: fmtPercent(trainer.metrics.validationAccuracy, 1) },
                      ]
                    : []),
                  { label: 'Parameters', value: trainer.network.parameterCount() },
                ]}
              />
            </div>
            {validation.length > 0 && trainer.bestEpoch ? (
              <div className="btn-row" style={{ marginTop: 10 }}>
                <Button onClick={trainer.restoreBest}>
                  Restore best val epoch ({trainer.bestEpoch.epoch})
                </Button>
              </div>
            ) : null}
          </Panel>

          <Panel title="Inspector">{inspector}</Panel>

          <Panel title="Data">
            <div className="stack stack--sm">
              <SelectField
                label="Dataset"
                value={datasetName}
                onChange={setDatasetName}
                options={DATASET_NAMES.map((name) => ({ value: name, label: DATASETS[name].label }))}
              />
              <Slider label="Points" min={40} max={500} step={20} value={count} onChange={setCount} display={count} />
              <Slider label="Noise σ" min={0} max={0.4} step={0.01} value={noise} onChange={setNoise} display={fmt(noise, 2)} />
              <Checkbox
                label="Hold out 25% for validation"
                checked={useValidation}
                onChange={setUseValidation}
              />
              <p className="faint" style={{ fontSize: 12.5, margin: 0 }}>
                {DATASETS[datasetName].requirement}
              </p>
            </div>
          </Panel>

          <Panel title="Architecture">
            <div className="stack stack--sm">
              <ArchitectureControls hiddenUnits={hiddenUnits} onChange={setHiddenUnits} maxLayers={4} maxUnits={12} />
              <Segmented
                label="Activation"
                value={activation}
                options={HIDDEN_ACTIVATIONS.map((n) => ({ value: n, label: ACTIVATIONS[n].label }))}
                onChange={setActivation}
              />
            </div>
          </Panel>

          <Panel title="Optimisation">
            <div className="stack stack--sm">
              <Slider
                label="Learning rate η"
                min={0.005}
                max={1.5}
                step={0.005}
                value={learningRate}
                onChange={setLearningRate}
                display={fmt(learningRate, 3)}
              />
              <Slider label="Batch size" min={1} max={64} step={1} value={batchSize} onChange={setBatchSize} display={batchSize} />
              <Slider label="L2 λ" min={0} max={0.1} step={0.001} value={l2} onChange={setL2} display={fmt(l2, 3)} />
              <Slider label="Dropout p" min={0} max={0.6} step={0.05} value={dropout} onChange={setDropout} display={fmt(dropout, 2)} />
              <Slider label="Epochs per frame" min={1} max={20} step={1} value={speed} onChange={setSpeed} display={speed} />
            </div>
          </Panel>
        </div>
      </div>

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">Experiments worth running</h3>
          <p>
            Each of these isolates one claim made earlier. The expected outcome is stated so that a
            surprise is informative rather than ambiguous.
          </p>

          <Detail kicker="section 02" title="A single neuron cannot solve XOR">
            <p>
              Dataset: XOR. Remove every hidden layer with <strong>− layer</strong>. Train.
            </p>
            <p>
              Accuracy stalls near 50% and the boundary is a straight line that cannot improve. The
              training loss plateaus well above zero — the failure is in the function family, not
              the optimiser. Add one hidden layer of 2 units and it solves the problem.
            </p>
          </Detail>

          <Detail kicker="section 04" title="Sigmoid hidden units stall in deep networks">
            <p>
              Dataset: Spiral, 4 hidden layers of 8 units, activation ReLU, η = 0.3. Train for 200
              epochs and note the loss. Switch the activation to Sigmoid and train again from the
              reset.
            </p>
            <p>
              The sigmoid network improves far more slowly at an equal number of epochs. Each of the
              four layers contributes a factor of at most 0.25 to the backward product, so the first
              layer receives a gradient roughly <M>{'10^{2}'}</M> times smaller than the last.
            </p>
          </Detail>

          <Detail kicker="section 07" title="The learning rate has a stability limit">
            <p>
              Dataset: Moons, 2 layers of 6, η = 0.05. Train — the loss falls smoothly. Now raise{' '}
              <M>{'\\eta'}</M> to 1.5 while it runs.
            </p>
            <p>
              The loss jumps and the boundary flails. Lower{' '}
              <M>{'\\eta'}</M> back to 0.1 without resetting and it recovers within a few epochs.
              The parameters were not destroyed; the steps were simply larger than the curvature
              allowed.
            </p>
          </Detail>

          <Detail kicker="section 09" title="Batch size trades noise against steps per epoch">
            <p>
              Dataset: Circles. Set the batch size to 1 and train for 30 epochs, then reset, set it
              to 64, and train for 30 epochs again.
            </p>
            <p>
              At <M>{'B = 1'}</M> the loss curve is visibly jagged but reaches a lower value per
              epoch, because it takes {'{'}points{'}'} updates per epoch instead of a handful. At{' '}
              <M>{'B = 64'}</M> the curve is smooth and progress per epoch is slower. Raising{' '}
              <M>{'\\eta'}</M> with <M>{'B'}</M> recovers most of the difference.
            </p>
          </Detail>

          <Detail kicker="section 10" title="Capacity without regularisation overfits">
            <p>
              Dataset: Moons, noise 0.35, 240 points, validation on. Set 4 layers of 12 units and
              train for several hundred epochs.
            </p>
            <p>
              The training loss keeps falling while the validation loss bottoms out and turns
              upward, and the boundary grows isolated pockets around individual training points. Now
              raise <M>{'\\lambda'}</M> to 0.02 without resetting: the pockets shrink within a few
              epochs and the validation loss falls again.
            </p>
          </Detail>

          <Detail kicker="section 11" title="Depth beats width at equal parameter count">
            <p>
              Dataset: Spiral, noise 0.05. Compare one hidden layer of 12 units (about 49
              parameters) with three hidden layers of 6 (about 103 parameters, but far more capable),
              then with two layers of 8. Train each for 300 epochs at η = 0.3.
            </p>
            <p>
              The deeper configurations reach a lower loss and a boundary that follows the spiral
              arms, while the single wide layer produces a boundary made of a few straight cuts.
            </p>
          </Detail>
        </div>

        <div className="prose-block">
          <h3 className="subhead">What the probe is for</h3>
          <p>
            The probe is a single input vector held fixed while everything else changes. Clicking a
            point on the plane sets <M>{'\\mathbf{x}'}</M>; the diagram then shows the exact
            activations that input produces at the current parameters, and the inspector shows the
            arithmetic behind any one of them.
          </p>
          <p>
            Three things it makes visible:
          </p>
          <ul>
            <li>
              <strong>What a hidden unit responds to.</strong> Move the probe across the boundary of
              one unit and watch its value cross zero. Each hidden unit has its own boundary, and
              the output layer combines them.
            </li>
            <li>
              <strong>Dead units.</strong> With ReLU, a unit that reads 0.00 wherever you put the
              probe is dead. Its incoming weights will never change again — select one of its
              connections and confirm the mean gradient is 0.
            </li>
            <li>
              <strong>Saturation.</strong> With tanh, a unit pinned at ±1 across the whole plane
              contributes almost no gradient, because <M>{"f'"}</M> is near zero there.
            </li>
          </ul>

          <h3 className="subhead">The full loop, in one place</h3>
          <p>
            Everything the earlier sections separated happens here simultaneously. One click on{' '}
            <strong>+1</strong> performs, for every mini-batch:
          </p>
          <ol>
            <li>
              A forward pass (section 05) producing <M>{'\\hat{y}'}</M> for each example, with
              dropout applied if <M>{'p > 0'}</M>.
            </li>
            <li>
              A loss evaluation (section 06) — binary cross-entropy against the labels.
            </li>
            <li>
              A backward pass (section 08) producing{' '}
              <M>{'\\partial L/\\partial W'}</M> and{' '}
              <M>{'\\partial L/\\partial b'}</M> for every parameter.
            </li>
            <li>
              A gradient-descent update (section 07), with the L2 term{' '}
              <M>{'\\lambda W'}</M> added to the gradient if{' '}
              <M>{'\\lambda > 0'}</M>.
            </li>
          </ol>
          <p>
            The decision boundary is then re-swept over a grid and redrawn, and the metrics are
            recomputed on the full training and validation sets with dropout disabled. Everything
            you see is produced by the same code the earlier sections stepped through by hand.
          </p>
        </div>
      </div>

      <Note title="Reading the inspector" accent>
        <p>
          The weight gradient shown is <M>{'\\partial J/\\partial w'}</M> averaged over the entire
          training set, which is the quantity full-batch gradient descent would use. During training
          the updates use mini-batch estimates of the same number, so individual steps differ from
          it by sampling noise. A weight whose mean gradient is near zero has stopped changing
          except through that noise and through the L2 term.
        </p>
      </Note>

      <Exercise
        notebook="12-capstone.ipynb"
        count={5}
        tasks={[
          <>Implement <code>__init__</code>, <code>forward</code>, <code>loss</code>, <code>gradients</code> and <code>fit</code></>,
          <>The check gradient-checks every parameter, then trains your network to 92% on circles</>,
          <>Run it on all five datasets and reproduce the playground’s probe readout</>,
        ]}
      >
        Build the entire library in one class, from an empty file.
      </Exercise>
    </Section>
  );
}
