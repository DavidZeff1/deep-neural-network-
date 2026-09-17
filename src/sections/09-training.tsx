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
import { Detail, InWords, Steps } from '../components/ui/Detail.tsx';
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
      <InWords>
        <p>
          Take a handful of training examples — that handful is called a batch, and{' '}
          <M>{'i \\in \\mathcal{B}'}</M> means "for each example <M>{'i'}</M> in it". Work out the
          gradient for each one, average them, and take one gradient-descent step using that average.
        </p>
        <p>
          Why average rather than use one example? One example's gradient points towards fitting that
          example, which is a noisy guess at what would help overall. Averaging a few cancels most of
          the noise. Why not use all of them? Because that costs a full pass over the dataset for a
          single step.
        </p>
      </InWords>

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
          <Steps>
            <li>Shuffle the training set and split it into batches of size <M>{'B'}</M>.</li>
            <li>
              For each batch: forward pass, loss, backward pass, then{' '}
              <M>{'\\theta \\leftarrow \\theta - \\eta\\, \\bar{g}'}</M> with{' '}
              <M>{'\\bar{g}'}</M> the mean gradient over the batch.
            </li>
            <li>After all batches, one epoch is complete; record the metrics.</li>
          </Steps>
          <p>
            With <M>{'m'}</M> examples and batch size <M>{'B'}</M>, one epoch performs{' '}
            <M>{'\\lceil m/B \\rceil'}</M> updates. The current settings — {train.length}{' '}
            examples at batch size {batchSize} — give{' '}
            <strong className="mono">{Math.ceil(train.length / batchSize)}</strong> updates per
            epoch. Halving the batch size doubles the number of updates per epoch while halving the
            work per update, so an epoch costs the same either way but moves the parameters through
            twice as many steps.
          </p>
          <p>
            The loss reported here is recomputed on the full training set after each epoch, with
            dropout disabled. This differs slightly from the running average frameworks print during
            an epoch, which is measured while the parameters are still changing and is therefore a
            mixture of several different models.
          </p>
          <p>
            The shuffle in step 1 matters. Without it the batches are the same every epoch, so the
            sequence of updates is periodic and the model can fit the batch boundaries rather than
            the data. If the file happens to be sorted by label, an unshuffled batch may contain a
            single class, and its gradient points towards predicting that class for everything.
          </p>

          <Detail kicker="derivation" title="Why the gradient noise falls as 1/√B">
            <p>
              Let <M>{'g_i = \\nabla_\\theta L(\\hat{y}^{(i)}, y^{(i)})'}</M> be the gradient
              from one example. The full-dataset gradient is the mean over all{' '}
              <M>{'m'}</M> examples; a batch estimate is the mean over{' '}
              <M>{'B'}</M> of them drawn uniformly:
            </p>
            <Equation plain>
              {'\\bar{g}_{\\mathcal{B}} = \\frac{1}{B}\\sum_{i\\in\\mathcal{B}} g_i'}
            </Equation>
            <p>
              Each draw has the full gradient as its expectation, so{' '}
              <M>{'\\mathbb{E}[\\bar{g}_{\\mathcal{B}}] = \\nabla J'}</M> — mini-batch
              gradients are unbiased. For the spread, use the variance of a mean of{' '}
              <M>{'B'}</M> independent draws:
            </p>
            <Equation plain>
              {'\\operatorname{Var}(\\bar{g}_{\\mathcal{B}}) = \\frac{1}{B^2}\\sum_{i\\in\\mathcal{B}}\\operatorname{Var}(g_i) = \\frac{\\Sigma}{B}'}
            </Equation>
            <InWords tag="the two steps">
              <p>
                First step: dividing by <M>{'B'}</M> divides the spread by{' '}
                <M>{'B'}</M> squared, because scaling a quantity by <M>{'c'}</M> scales its variance
                by <M>{'c^2'}</M> (section 00).
              </p>
              <p>
                Second step: the <M>{'B'}</M> independent examples each contribute the same amount of
                spread <M>{'\\Sigma'}</M>, and independent contributions add, so the sum is{' '}
                <M>{'B\\Sigma'}</M>. Combining, <M>{'B\\Sigma / B^2 = \\Sigma/B'}</M>.
              </p>
            </InWords>
            <p>
              where <M>{'\\Sigma'}</M> is the per-example gradient covariance. The standard
              deviation is therefore <M>{'\\sqrt{\\Sigma/B}'}</M>, proportional to{' '}
              <M>{'1/\\sqrt{B}'}</M>.
            </p>
            <p>
              The consequence is a diminishing return. Going from <M>{'B = 1'}</M> to{' '}
              <M>{'B = 4'}</M> halves the noise at four times the cost per update. Going from{' '}
              <M>{'B = 64'}</M> to <M>{'B = 256'}</M> halves it again, also at four times the cost —
              but by then the noise is already small relative to the gradient, so the extra
              precision buys very little. This is why batch sizes cluster in the tens to low
              hundreds rather than being made as large as memory allows.
            </p>
            <p>
              Set the batch size to 1 above and the loss curve becomes visibly jagged; set it to 64
              and the curve smooths out while progress per epoch often slows. Neither is a bug —
              they are the two ends of the same trade.
            </p>
          </Detail>
        </div>
        <div className="prose-block">
          <h3 className="subhead">Learning rate and batch size are coupled</h3>
          <p>
            Over one epoch, SGD takes <M>{'m/B'}</M> steps of size proportional to{' '}
            <M>{'\\eta'}</M>, so the total distance travelled scales as{' '}
            <M>{'\\eta m / B'}</M>. Halving <M>{'B'}</M> doubles the number of steps and therefore
            doubles the distance covered per epoch at fixed <M>{'\\eta'}</M>.
          </p>
          <p>
            The common heuristic — the <em>linear scaling rule</em> — is to change{' '}
            <M>{'\\eta'}</M> in proportion to <M>{'B'}</M>, keeping{' '}
            <M>{'\\eta/B'}</M> fixed. That preserves both the per-epoch distance and the ratio of
            gradient noise to step size, which together determine how far the iterates wander around
            a minimum. It is a good approximation for moderate batch sizes and breaks down at very
            large ones, where the noise is already negligible and the rule pushes{' '}
            <M>{'\\eta'}</M> past the stability bound from section 07.
          </p>
          <p>
            The practical form: if you double the batch size and training slows down, double the
            learning rate before concluding the larger batch is worse.
          </p>

          <h3 className="subhead">Reading the curves</h3>
          <table className="data" style={{ marginBottom: 14 }}>
            <thead>
              <tr>
                <th>What you see</th>
                <th>Most likely cause</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Loss rises or oscillates wildly</td>
                <td>η above the stability limit</td>
              </tr>
              <tr>
                <td>Loss falls, then plateaus high</td>
                <td>Model too small, or dead units</td>
              </tr>
              <tr>
                <td>Loss barely moves from the start</td>
                <td>η far too small, or vanishing gradients</td>
              </tr>
              <tr>
                <td>Loss falls smoothly but very slowly</td>
                <td>η small, or a badly conditioned surface</td>
              </tr>
              <tr>
                <td>Loss becomes NaN</td>
                <td>Overflow — η far too large, or log(0)</td>
              </tr>
              <tr>
                <td>Jagged but descending</td>
                <td>Normal for a small batch size</td>
              </tr>
            </tbody>
          </table>
          <p>
            A useful first move for any of the top three: reduce <M>{'\\eta'}</M> by a factor of
            ten and see which symptom changes. It separates optimisation problems from capacity
            problems in one experiment.
          </p>

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

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">When is training finished?</h3>
          <p>
            There is no state the algorithm reaches and reports. Gradient descent on a non-convex
            surface with stochastic gradients does not converge to a point; it settles into a region
            and wanders inside it, with the size of the region set by{' '}
            <M>{'\\eta'}</M> and the gradient noise. Training stops when a criterion you choose is
            met, not when the algorithm says so.
          </p>
          <p>The criteria actually used, in rough order of preference:</p>
          <ul>
            <li>
              <strong>Validation loss stops improving</strong> for a fixed number of epochs — early
              stopping, covered in section 10. This is the standard.
            </li>
            <li>
              <strong>A compute budget is exhausted.</strong> Honest, and common at scale.
            </li>
            <li>
              <strong>The training loss falls below a threshold.</strong> Only meaningful when the
              threshold comes from the noise floor of the data rather than from a round number.
            </li>
          </ul>
          <p>
            A training loss of exactly zero is usually a warning rather than an achievement: it
            means the model has enough capacity to fit every training label exactly, including the
            mislabelled ones.
          </p>
        </div>
        <div className="prose-block">
          <h3 className="subhead">What stochasticity buys</h3>
          <p>
            The noise in mini-batch gradients is not purely a cost. Full-batch gradient descent on a
            non-convex surface follows the slope exactly and stops at the first stationary point it
            reaches — including saddle points and sharp, narrow minima. Mini-batch noise perturbs
            each step, which lets the iterates escape both.
          </p>
          <p>
            The effect is selective. A minimum that is narrow in some direction has high curvature
            there, so noise of a given size produces a large increase in loss and the iterates get
            pushed out. A wide, flat minimum absorbs the same noise with little change in loss and
            the iterates stay. SGD therefore drifts towards flatter minima, and flatter minima tend
            to generalise better — a small change in the parameters, or in the data, changes the
            predictions less.
          </p>
          <p>
            This is one reason very large batches can generalise worse than moderate ones at an
            equal number of epochs, even when the training loss reached is the same.
          </p>
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
