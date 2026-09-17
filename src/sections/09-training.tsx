import { useEffect, useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig } from '../lib/network.ts';
import { DATASETS, DATASET_NAMES, generateDataset } from '../lib/datasets.ts';
import type { DatasetName } from '../lib/datasets.ts';
import { ACTIVATIONS, HIDDEN_ACTIVATIONS } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { useTrainer } from '../hooks/useTrainer.ts';
import { NetworkDiagram } from '../components/viz/NetworkDiagram.tsx';
import { DecisionBoundary } from '../components/viz/DecisionBoundary.tsx';
import { MetricChart } from '../components/viz/MetricChart.tsx';
import { ArchitectureControls } from '../components/ui/ArchitectureControls.tsx';
import { Panel, Note, Stats, Legend } from '../components/ui/layout.tsx';
import { Button, SelectField, Segmented, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { fmt, fmtPercent } from '../lib/format.ts';

const LOSS_COLOR = '#3b82f6';
const ACC_COLOR = '#16a34a';

export function TrainingSection({ id, index }: SectionProps) {
  const [datasetName, setDatasetName] = useState<DatasetName>('circles');
  const [noise, setNoise] = useState(0.08);
  const [count, setCount] = useState(200);
  const [hiddenUnits, setHiddenUnits] = useState<number[]>([6, 6]);
  const [activation, setActivation] = useState<ActivationName>('tanh');
  const [learningRate, setLearningRate] = useState(0.2);
  const [batchSize, setBatchSize] = useState(16);
  const [speed, setSpeed] = useState(3);

  const train = useMemo(
    () => generateDataset(datasetName, { count, noise, seed: 17 }),
    [datasetName, count, noise],
  );

  const config = useMemo<NetworkConfig>(
    () => ({
      inputSize: 2,
      hiddenUnits,
      outputSize: 1,
      hiddenActivation: activation,
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 77,
    }),
    [hiddenUnits, activation],
  );

  const trainer = useTrainer({
    config,
    train,
    learningRate,
    batchSize,
    epochsPerTick: speed,
    maxEpochs: 3000,
  });

  // Pause when the tab is hidden so a forgotten run does not spin in the background.
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden && trainer.running) trainer.pause();
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [trainer]);

  const predict = useMemo(
    () => (x1: number, x2: number) => trainer.network.forward([x1, x2]).output[0],
    // `version` is the dependency that matters: the network is mutated in place.
    [trainer.network, trainer.version],
  );

  const lossSeries = useMemo(
    () => [
      {
        label: 'training loss',
        color: LOSS_COLOR,
        values: trainer.history.map((p) => [p.epoch, p.trainLoss] as [number, number]),
      },
    ],
    [trainer.history],
  );

  const accuracySeries = useMemo(
    () => [
      {
        label: 'training accuracy',
        color: ACC_COLOR,
        values: trainer.history.map((p) => [p.epoch, p.trainAccuracy] as [number, number]),
      },
    ],
    [trainer.history],
  );

  const meta = DATASETS[datasetName];

  return (
    <Section
      id={id}
      index={index}
      title="Training"
      lede={
        <>
          Training repeats one loop: take a mini-batch, run the forward pass, compute the loss,
          backpropagate to get every gradient, and update every parameter. One pass over the whole
          dataset is an epoch.
        </>
      }
    >
      <Equation caption="B is the batch size. Each update uses the average gradient over the batch.">
        {'\\theta \\leftarrow \\theta - \\eta \\cdot \\frac{1}{B}\\sum_{i \\in \\mathcal{B}} \\nabla_{\\theta} L\\!\\left(\\hat{y}^{(i)}, y^{(i)}\\right)'}
      </Equation>

      <div className="grid grid--side">
        <div className="stack">
          <Panel
            title="Decision boundary"
            hint={trainer.running ? 'training…' : `epoch ${trainer.metrics.epoch}`}
            caption="Background shading is the network's output probability across the input plane; the dark curve is where it equals 0.5. Points are the training set."
          >
            <div className="grid grid--canvas" style={{ gap: 16 }}>
              <DecisionBoundary
                predict={predict}
                train={train}
                version={trainer.version}
                resolution={64}
                hint="x₁ →, x₂ ↑"
              />
              <div className="stack stack--sm">
                <NetworkDiagram
                  network={trainer.network}
                  height={210}
                  showValues={false}
                  inputLabels={['x₁', 'x₂']}
                  outputLabels={['ŷ']}
                />
                <Legend
                  items={[
                    { color: '#3b82f6', label: 'class 1 / positive weight' },
                    { color: '#e0761f', label: 'class 0 / negative weight' },
                  ]}
                />
                <p className="faint" style={{ fontSize: 12.5, margin: 0 }}>
                  {meta.description} {meta.requirement}
                </p>
              </div>
            </div>
          </Panel>

          <div className="grid grid--2">
            <Panel title="Training loss" flush>
              <div style={{ padding: '10px 10px 0' }}>
                <MetricChart series={lossSeries} yMin={0} yLabel="loss" ariaLabel="Training loss curve" />
              </div>
            </Panel>
            <Panel title="Training accuracy" flush>
              <div style={{ padding: '10px 10px 0' }}>
                <MetricChart
                  series={accuracySeries}
                  yMin={0}
                  yMax={1.02}
                  yLabel="accuracy"
                  ariaLabel="Training accuracy curve"
                />
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
              <Button onClick={() => trainer.runEpochs(1)}>+1 epoch</Button>
              <Button onClick={() => trainer.runEpochs(10)}>+10</Button>
              <Button onClick={() => trainer.reset()}>Reset</Button>
              <Button onClick={() => trainer.reset(Math.floor(Math.random() * 100000))}>New init</Button>
            </div>
            <div style={{ marginTop: 14 }}>
              <Stats
                items={[
                  { label: 'Epoch', value: trainer.metrics.epoch },
                  { label: 'Loss', value: fmt(trainer.metrics.trainLoss, 4), accent: true },
                  { label: 'Accuracy', value: fmtPercent(trainer.metrics.trainAccuracy, 1) },
                  { label: 'Parameters', value: trainer.network.parameterCount() },
                ]}
              />
            </div>
          </Panel>

          <Panel title="Data">
            <div className="stack stack--sm">
              <SelectField
                label="Dataset"
                value={datasetName}
                onChange={setDatasetName}
                options={DATASET_NAMES.map((name) => ({ value: name, label: DATASETS[name].label }))}
              />
              <Slider label="Points" min={40} max={400} step={20} value={count} onChange={setCount} display={count} />
              <Slider label="Noise σ" min={0} max={0.3} step={0.01} value={noise} onChange={setNoise} display={fmt(noise, 2)} />
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
              <Slider
                label="Batch size"
                min={1}
                max={64}
                step={1}
                value={batchSize}
                onChange={setBatchSize}
                display={batchSize}
              />
              <Slider
                label="Epochs per frame"
                min={1}
                max={20}
                step={1}
                value={speed}
                onChange={setSpeed}
                display={speed}
              />
            </div>
            <p className="faint" style={{ fontSize: 12.5, marginTop: 10, marginBottom: 0 }}>
              Learning rate and batch size take effect immediately — change them while training runs.
            </p>
          </Panel>
        </div>
      </div>

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">The loop, precisely</h3>
          <ol>
            <li>Shuffle the training set and split it into batches of size <M>{'B'}</M>.</li>
            <li>
              For each batch: forward pass, loss, backward pass, then{' '}
              <M>{'\\theta \\leftarrow \\theta - \\eta\\, \\bar{g}'}</M> with{' '}
              <M>{'\\bar{g}'}</M> the mean gradient over the batch.
            </li>
            <li>After all batches, one epoch is complete; record the metrics.</li>
          </ol>
          <p>
            The loss reported here is recomputed on the full training set after each epoch, with
            dropout disabled. This differs slightly from the running average frameworks print during
            an epoch, which is measured while the parameters are still changing.
          </p>
        </div>
        <div className="prose-block">
          <h3 className="subhead">What to try</h3>
          <ul>
            <li>
              <strong>XOR with no hidden layer:</strong> remove both hidden layers. Accuracy stalls
              near 50% — a single linear boundary cannot separate opposite quadrants.
            </li>
            <li>
              <strong>Learning rate 1.5 on Spiral:</strong> the loss jumps and the boundary flails.
              Reduce to 0.1 and it settles.
            </li>
            <li>
              <strong>Batch size 1:</strong> each update uses one example, so the loss curve becomes
              visibly noisy, but progress per epoch is often faster early on.
            </li>
            <li>
              <strong>Sigmoid hidden units with 4 layers:</strong> training barely moves, because{' '}
              <M>{"\\sigma' \\le 0.25"}</M> shrinks the gradient at every layer.
            </li>
          </ul>
        </div>
      </div>

      <Note title="Accuracy and loss measure different things" accent>
        <p>
          Accuracy counts predictions on the correct side of 0.5. Loss measures how confident those
          predictions are. Accuracy can sit at 100% while the loss keeps falling, as the network
          pushes outputs from 0.6 towards 0.99. Accuracy is also a step function of the parameters,
          so it has zero gradient almost everywhere — which is why the loss, not the accuracy, is
          what gets minimised.
        </p>
      </Note>
    </Section>
  );
}
