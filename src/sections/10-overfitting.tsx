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
import { Detail } from '../components/ui/Detail.tsx';
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

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">What overfitting is, stated precisely</h3>
          <p>
            Training minimises the empirical risk — the average loss over the sample you have. What
            you actually want is the expected loss over the distribution the sample came from:
          </p>
          <Equation plain>
            {'\\hat{R}(\\theta) = \\frac{1}{m}\\sum_{i=1}^{m} L\\!\\left(f_\\theta(\\mathbf{x}^{(i)}), y^{(i)}\\right) \\qquad\\text{vs}\\qquad R(\\theta) = \\mathbb{E}_{(\\mathbf{x},y)}\\bigl[L(f_\\theta(\\mathbf{x}), y)\\bigr]'}
          </Equation>
          <p>
            The gap <M>{'R(\\theta) - \\hat{R}(\\theta)'}</M> is the generalisation gap.
            Overfitting is the situation where minimising{' '}
            <M>{'\\hat{R}'}</M> further <em>increases</em> <M>{'R'}</M>: the model is using its
            remaining capacity to fit features of this particular sample — its noise — that do not
            recur.
          </p>
          <p>
            <M>{'R'}</M> cannot be computed. The validation loss is an unbiased estimate of it,
            valid exactly as long as the validation set has not influenced any choice made about the
            model.
          </p>

          <h3 className="subhead">Bias and variance</h3>
          <p>
            For squared error the gap decomposes exactly. Let{' '}
            <M>{'y = h(\\mathbf{x}) + \\varepsilon'}</M> with{' '}
            <M>{'\\mathbb{E}[\\varepsilon] = 0'}</M> and{' '}
            <M>{'\\operatorname{Var}(\\varepsilon) = \\sigma^2'}</M>, and let{' '}
            <M>{'f_D'}</M> be the model fitted on a random training set <M>{'D'}</M>. Then at any
            fixed <M>{'\\mathbf{x}'}</M>:
          </p>
          <Equation plain>
            {'\\mathbb{E}_{D,\\varepsilon}\\bigl[(y - f_D(\\mathbf{x}))^2\\bigr] = \\underbrace{\\bigl(h(\\mathbf{x}) - \\bar{f}(\\mathbf{x})\\bigr)^2}_{\\text{bias}^2} + \\underbrace{\\mathbb{E}_D\\bigl[(f_D(\\mathbf{x})-\\bar{f}(\\mathbf{x}))^2\\bigr]}_{\\text{variance}} + \\underbrace{\\sigma^2}_{\\text{noise}}'}
          </Equation>
          <p>
            with <M>{'\\bar{f} = \\mathbb{E}_D[f_D]'}</M>. Bias is the error of the average model
            — how far the family can get from the truth at all. Variance is how much the fitted
            model moves when the training sample changes. Noise is irreducible: no model can do
            better than <M>{'\\sigma^2'}</M>.
          </p>
          <p>
            Increasing capacity lowers bias and raises variance. Regularisation does the reverse.
            The three techniques below are three ways of buying a reduction in variance for an
            increase in bias, and each is worthwhile exactly while the variance saved exceeds the
            bias added.
          </p>
          <p>
            You can watch the variance term directly: press <strong>Reset</strong> a few times with
            40 units and 3 layers. Each run starts from a different initialisation and converges to
            a visibly different boundary while fitting the same points. That spread{' '}
            <em>is</em> the variance.
          </p>

          <Detail title="Why more parameters than examples is not automatically fatal">
            <p>
              The classical account says a model with more parameters than training examples can fit
              any labelling and must therefore generalise badly. Networks routinely violate this:
              the panel above has {trainer.network.parameterCount()} parameters fitted to{' '}
              {train.length} points and can still reach a reasonable validation loss.
            </p>
            <p>
              The resolution is that the parameter count is not the effective capacity. What
              matters is the size of the set of functions the training procedure actually reaches,
              and that is much smaller than the set the architecture could represent. Gradient
              descent from a small random initialisation, with a finite number of steps and a finite
              learning rate, has strong implicit biases: it tends to find low-norm, low-curvature
              solutions long before it finds the wild ones that also fit the data.
            </p>
            <p>
              Empirically, test error as a function of model size often falls, rises around the point
              where the model can just interpolate the training set, then falls again as the model
              grows further — the "double descent" curve. The practical message is narrow and worth
              stating plainly: parameter count alone predicts very little, and the validation curve
              is the measurement that matters.
            </p>
          </Detail>
        </div>

        <div className="prose-block">
          <h3 className="subhead">Train, validation, test</h3>
          <p>
            Three splits, three distinct jobs:
          </p>
          <ul>
            <li>
              <strong>Training set</strong> — the only data that contributes gradients.
            </li>
            <li>
              <strong>Validation set</strong> — used to choose hyperparameters: architecture,{' '}
              <M>{'\\lambda'}</M>, the stopping epoch. Never differentiated through, but heavily
              selected on.
            </li>
            <li>
              <strong>Test set</strong> — looked at once, at the end, to estimate performance.
            </li>
          </ul>
          <p>
            The reason for the third split is that selection is a form of fitting. If you try 50
            configurations and keep the one with the lowest validation loss, that minimum is biased
            downward — you have partly fitted the validation set through the choice. With 50
            independent attempts, the best of them beats its true mean by roughly two standard
            deviations of the validation estimate by chance alone.
          </p>
          <p>
            Early stopping is the clearest case: the epoch selected is the argmin of a noisy curve
            measured on the validation set, so the validation loss at that epoch is an optimistic
            estimate of the true loss by construction. It is still the right epoch to pick; it is
            just not an honest number to report.
          </p>
          <p>
            This page has no test set, because nothing here is being reported as a final result. A
            real experiment needs one.
          </p>

          <h3 className="subhead">Reading the two curves</h3>
          <table className="data" style={{ marginBottom: 14 }}>
            <thead>
              <tr>
                <th>Training</th>
                <th>Validation</th>
                <th>Diagnosis</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>high</td>
                <td>high, close to training</td>
                <td>Underfitting — add capacity or train longer</td>
              </tr>
              <tr>
                <td>low</td>
                <td>low, close to training</td>
                <td>Fitting well</td>
              </tr>
              <tr>
                <td>low</td>
                <td>higher and rising</td>
                <td>Overfitting — regularise or stop earlier</td>
              </tr>
              <tr>
                <td>low</td>
                <td>much higher, flat</td>
                <td>Overfitting, or the splits differ in distribution</td>
              </tr>
              <tr>
                <td>high</td>
                <td>lower than training</td>
                <td>Usually dropout: it is active in training, not in evaluation</td>
              </tr>
            </tbody>
          </table>
          <p>
            The last row is worth knowing before it puzzles you. With dropout at{' '}
            <M>{'p = 0.5'}</M> the training loss is measured on a network with half its units
            removed, while the validation loss uses the full network. The validation loss being
            lower is the expected behaviour, not a bug.
          </p>
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
          <p>
            Why penalising the norm reduces variance: with the activations fixed, the output's
            sensitivity to a change in the input is bounded by the product of the weight-matrix
            norms. Shrinking those norms forces a smoother function — smaller derivatives, gentler
            boundaries — and smooth functions vary less when the training sample changes.
          </p>
          <Detail title="L2 is a Gaussian prior on the weights">
            <p>
              Put a zero-mean Gaussian prior on every weight,{' '}
              <M>{'w \\sim \\mathcal{N}(0, \\tau^2)'}</M>, and take the maximum a-posteriori
              estimate instead of the maximum-likelihood one:
            </p>
            <Equation plain>
              {'\\theta_{\\text{MAP}} = \\arg\\max_\\theta \\; \\log P(D\\mid\\theta) + \\log P(\\theta)'}
            </Equation>
            <p>
              The prior term is{' '}
              <M>{'\\log P(\\theta) = -\\frac{1}{2\\tau^2}\\sum_k w_k^2 + \\text{const}'}</M>.
              Negating to turn the maximisation into a minimisation gives exactly the L2 penalty
              with <M>{'\\lambda = 1/\\tau^2'}</M>.
            </p>
            <p>
              So <M>{'\\lambda'}</M> encodes a belief about the scale of the weights. A large{' '}
              <M>{'\\lambda'}</M> is a narrow prior — a strong prior belief that the weights are
              small — and it takes correspondingly more evidence in the data to move them. This is
              also why the biases are left out: a prior pulling them towards zero would encode a
              belief that the function passes through the origin, which is rarely intended.
            </p>
          </Detail>
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
          <Detail title="Dropout as an ensemble">
            <p>
              A layer of <M>{'n'}</M> units has <M>{'2^{n}'}</M> possible masks, and each training
              step samples one of them. Over training, the procedure is fitting an enormous
              collection of thinned networks that share all their weights.
            </p>
            <p>
              At evaluation time, averaging the predictions of all <M>{'2^{n}'}</M> of them is
              impossible for <M>{'n'}</M> beyond about 20. The inverted-dropout rescaling is an
              approximation to that average: because{' '}
              <M>{'\\mathbb{E}[m_j/(1-p)] = 1'}</M>, the expected input to the next layer matches
              the full network's input. For a linear unit this approximation is exact; for a
              non-linear one it is not, but it is close enough to work and costs one forward pass
              instead of <M>{'2^{n}'}</M>.
            </p>
            <p>
              The ensemble view also predicts the correct failure mode. Averaging many models
              reduces variance, not bias — so dropout helps a model that is overfitting and hurts
              one that is already underfitting, by removing capacity it did not have to spare. Set
              the width to 4 and dropout to 0.5 above and the training loss stops falling at all.
            </p>
          </Detail>
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
          <p>
            In practice it is run with <em>patience</em>: keep training until the validation loss has
            failed to improve for <M>{'k'}</M> consecutive epochs, then restore the best
            parameters. Without patience a single noisy epoch stops training prematurely.
          </p>
          <p>
            There is a connection to L2 worth noticing. Gradient descent from a small
            initialisation reaches large-norm solutions only after many steps, so stopping early
            bounds the distance travelled from the origin — which bounds the weight norm. On a
            quadratic objective the correspondence is exact: stopping after{' '}
            <M>{'t'}</M> steps with rate <M>{'\\eta'}</M> gives the same solution as L2 with{' '}
            <M>{'\\lambda \\approx 1/(\\eta t)'}</M>. Fewer epochs is stronger
            regularisation.
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
