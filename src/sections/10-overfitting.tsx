import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import type { NetworkConfig } from '../lib/network.ts';
import { DATASETS, DATASET_NAMES, generateDataset, splitDataset } from '../lib/datasets.ts';
import type { DatasetName } from '../lib/datasets.ts';
import { useTrainer } from '../hooks/useTrainer.ts';
import { DecisionBoundary } from '../components/viz/DecisionBoundary.tsx';
import { MetricChart } from '../components/viz/MetricChart.tsx';
import { Panel, Note, Stats, Legend } from '../components/ui/layout.tsx';
import { Button, SelectField, Slider } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { fmt, fmtPercent } from '../lib/format.ts';

const TRAIN_COLOR = '#3b82f6';
const VAL_COLOR = '#e0761f';

export function OverfittingSection({ id, index }: SectionProps) {
  const [datasetName, setDatasetName] = useState<DatasetName>('moons');
  const [width, setWidth] = useState(24);
  const [depth, setDepth] = useState(2);
  const [l2, setL2] = useState(0);
  const [dropout, setDropout] = useState(0);
  const [pointCount, setPointCount] = useState(80);
  const [noise, setNoise] = useState(0.22);

  const { train, validation } = useMemo(() => {
    const all = generateDataset(datasetName, { count: pointCount * 2, noise, seed: 91 });
    return splitDataset(all, 0.5, 5);
  }, [datasetName, pointCount, noise]);

  const config = useMemo<NetworkConfig>(
    () => ({
      inputSize: 2,
      hiddenUnits: Array.from({ length: depth }, () => width),
      outputSize: 1,
      hiddenActivation: 'relu',
      outputActivation: 'sigmoid',
      loss: 'bce',
      seed: 404,
      l2,
      dropout,
    }),
    [depth, width, l2, dropout],
  );

  const trainer = useTrainer({
    config,
    train,
    validation,
    learningRate: 0.08,
    batchSize: 8,
    epochsPerTick: 2,
    maxEpochs: 1500,
  });

  const predict = useMemo(
    () => (x1: number, x2: number) => trainer.network.forward([x1, x2]).output[0],
    // `version` is the dependency that matters: the network is mutated in place.
    [trainer.network, trainer.version],
  );

  const lossSeries = useMemo(
    () => [
      {
        label: 'training',
        color: TRAIN_COLOR,
        values: trainer.history.map((p) => [p.epoch, p.trainLoss] as [number, number]),
      },
      {
        label: 'validation',
        color: VAL_COLOR,
        dash: '5 4',
        values: trainer.history.map((p) => [p.epoch, p.validationLoss] as [number, number]),
      },
    ],
    [trainer.history],
  );

  const accuracySeries = useMemo(
    () => [
      {
        label: 'training',
        color: TRAIN_COLOR,
        values: trainer.history.map((p) => [p.epoch, p.trainAccuracy] as [number, number]),
      },
      {
        label: 'validation',
        color: VAL_COLOR,
        dash: '5 4',
        values: trainer.history.map((p) => [p.epoch, p.validationAccuracy] as [number, number]),
      },
    ],
    [trainer.history],
  );

  const gap = trainer.metrics.validationLoss - trainer.metrics.trainLoss;
  const l2Penalty = trainer.network.l2Penalty();

  return (
    <Section
      id={id}
      index={index}
      title="Overfitting & regularisation"
      lede={
        <>
          A network with enough parameters can drive the training loss to nearly zero by memorising
          the training points, including their noise. The validation loss — measured on data the
          updates never touched — is what reveals it.
        </>
      }
    >
      <div className="grid grid--side">
        <div className="stack">
          <Panel
            title="Decision boundary"
            hint={`epoch ${trainer.metrics.epoch}`}
            caption="Filled circles are training points; hollow circles are validation points, which never contribute a gradient. A boundary that bends around individual filled points, isolating hollow ones on the wrong side, is overfitting."
          >
            <div className="grid grid--canvas" style={{ gap: 16 }}>
              <DecisionBoundary
                predict={predict}
                train={train}
                validation={validation}
                version={trainer.version}
                resolution={64}
              />
              <div className="stack stack--sm">
                <Stats
                  items={[
                    { label: 'Train loss', value: fmt(trainer.metrics.trainLoss, 4) },
                    { label: 'Val loss', value: fmt(trainer.metrics.validationLoss, 4), accent: true },
                    { label: 'Gap', value: fmt(gap, 4) },
                    { label: 'Train acc', value: fmtPercent(trainer.metrics.trainAccuracy, 1) },
                    { label: 'Val acc', value: fmtPercent(trainer.metrics.validationAccuracy, 1) },
                    { label: 'Parameters', value: trainer.network.parameterCount() },
                  ]}
                />
                <Legend
                  items={[
                    { color: TRAIN_COLOR, label: 'class 1' },
                    { color: VAL_COLOR, label: 'class 0' },
                    { color: 'var(--text-faint)', label: 'hollow = validation', outline: true },
                  ]}
                />
                <p className="faint" style={{ fontSize: 12.5, margin: 0 }}>
                  {trainer.network.parameterCount()} parameters fitted to {train.length} training
                  points. When parameters outnumber examples, an exact fit to the training set is
                  always available.
                </p>
              </div>
            </div>
          </Panel>

          <div className="grid grid--2">
            <Panel
              title="Loss"
              flush
              caption={
                <Legend
                  items={[
                    { color: TRAIN_COLOR, label: 'training' },
                    { color: VAL_COLOR, label: 'validation', dashed: true },
                  ]}
                />
              }
            >
              <div style={{ padding: '10px 10px 0' }}>
                <MetricChart
                  series={lossSeries}
                  yMin={0}
                  yLabel="loss"
                  markerX={trainer.bestEpoch?.epoch ?? null}
                  markerLabel="best val"
                  ariaLabel="Training and validation loss"
                />
              </div>
            </Panel>
            <Panel title="Accuracy" flush>
              <div style={{ padding: '10px 10px 0' }}>
                <MetricChart
                  series={accuracySeries}
                  yMin={0}
                  yMax={1.02}
                  yLabel="accuracy"
                  ariaLabel="Training and validation accuracy"
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
              <Button onClick={() => trainer.runEpochs(20)}>+20 epochs</Button>
              <Button onClick={() => trainer.reset()}>Reset</Button>
            </div>
            <div className="btn-row" style={{ marginTop: 10 }}>
              <Button onClick={trainer.restoreBest} disabled={!trainer.bestEpoch}>
                Early stop: restore epoch {trainer.bestEpoch?.epoch ?? '—'}
              </Button>
            </div>
            {trainer.bestEpoch ? (
              <p className="faint" style={{ fontSize: 12.5, marginTop: 10, marginBottom: 0 }}>
                Lowest validation loss so far: {fmt(trainer.bestEpoch.validationLoss, 4)} at epoch{' '}
                {trainer.bestEpoch.epoch}. Current: {fmt(trainer.metrics.validationLoss, 4)}.
              </p>
            ) : null}
          </Panel>

          <Panel title="Model complexity">
            <div className="stack stack--sm">
              <Slider label="Units per layer" min={1} max={40} step={1} value={width} onChange={setWidth} display={width} />
              <Slider label="Hidden layers" min={1} max={4} step={1} value={depth} onChange={setDepth} display={depth} />
            </div>
          </Panel>

          <Panel title="Regularisation">
            <div className="stack stack--sm">
              <Slider
                label={<>L2 coefficient λ</>}
                min={0}
                max={0.2}
                step={0.001}
                value={l2}
                onChange={setL2}
                display={fmt(l2, 3)}
              />
              <Slider
                label="Dropout rate p"
                min={0}
                max={0.7}
                step={0.05}
                value={dropout}
                onChange={setDropout}
                display={fmt(dropout, 2)}
              />
            </div>
            <p className="faint" style={{ fontSize: 12.5, marginTop: 10, marginBottom: 0 }}>
              Both take effect on the next epoch without resetting the weights. Current L2 penalty
              term: <span className="mono">{fmt(l2Penalty, 4)}</span>.
            </p>
          </Panel>

          <Panel title="Data">
            <div className="stack stack--sm">
              <SelectField
                label="Dataset"
                value={datasetName}
                onChange={setDatasetName}
                options={DATASET_NAMES.map((name) => ({ value: name, label: DATASETS[name].label }))}
              />
              <Slider
                label="Points per split"
                min={20}
                max={200}
                step={10}
                value={pointCount}
                onChange={setPointCount}
                display={pointCount}
              />
              <Slider label="Noise σ" min={0} max={0.4} step={0.01} value={noise} onChange={setNoise} display={fmt(noise, 2)} />
            </div>
          </Panel>
        </div>
      </div>

      <div className="grid grid--3">
        <div className="prose-block">
          <h3 className="subhead">L2 regularisation</h3>
          <p>Add the squared norm of the weights to the objective:</p>
          <Equation plain>
            {'J = \\frac{1}{m}\\sum_i L_i + \\frac{\\lambda}{2}\\sum_{l} \\lVert W^{(l)} \\rVert_F^2'}
          </Equation>
          <p>
            The extra term contributes <M>{'\\lambda W'}</M> to the gradient, so the update becomes
          </p>
          <Equation plain>
            {'W \\leftarrow (1 - \\eta\\lambda)\\,W - \\eta\\, \\nabla_W L'}
          </Equation>
          <p>
            Every weight is multiplied by a factor slightly below 1 at each step — hence the name
            weight decay. Biases are usually excluded: they shift the function without increasing
            its curvature.
          </p>
        </div>

        <div className="prose-block">
          <h3 className="subhead">Dropout</h3>
          <p>
            During training each hidden unit is set to zero with probability <M>{'p'}</M>,
            independently per example. Surviving activations are divided by <M>{'1-p'}</M> so the
            expected value entering the next layer is unchanged:
          </p>
          <Equation plain>
            {'\\tilde{a}_j = \\frac{m_j}{1-p}\\,a_j, \\quad m_j \\sim \\mathrm{Bernoulli}(1-p)'}
          </Equation>
          <p>
            At evaluation time no units are dropped and no rescaling is applied — this is inverted
            dropout, the standard implementation. Because a unit cannot rely on any particular
            other unit being present, the network is pushed towards redundant representations.
          </p>
        </div>

        <div className="prose-block">
          <h3 className="subhead">Early stopping</h3>
          <p>
            Track the validation loss each epoch and keep the parameters from the epoch where it was
            lowest. Training past that point continues to reduce the training loss while the
            validation loss rises: the extra capacity is being spent on noise.
          </p>
          <p>
            The dashed vertical line on the loss chart marks that epoch. The button restores those
            parameters, and the boundary visibly simplifies.
          </p>
          <p>
            Early stopping needs no change to the objective, which makes it the cheapest of the
            three — but it consumes a validation set, and the epoch it selects is itself fitted to
            that set.
          </p>
        </div>
      </div>

      <Note title="A sequence worth running" accent>
        <p>
          Set 40 units, 3 layers, λ = 0, dropout = 0, and train for a few hundred epochs on Moons
          with σ = 0.25. The training loss approaches zero while the validation loss turns upward,
          and the boundary grows isolated pockets around individual training points. Now raise λ to
          0.02 without resetting: within a few epochs the pockets shrink and the validation loss
          falls. Raise λ to 0.2 and the boundary becomes nearly linear — the penalty now dominates
          the data term, which is underfitting.
        </p>
      </Note>
    </Section>
  );
}
