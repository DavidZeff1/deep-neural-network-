import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import { LOSSES } from '../lib/losses.ts';
import type { LossName } from '../lib/losses.ts';
import { softmax } from '../lib/activations.ts';
import { Panel, Note, Stats, Legend } from '../components/ui/layout.tsx';
import { Segmented, Slider, Button } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { Curve, Marker, Plot } from '../components/viz/Plot.tsx';
import { fmt } from '../lib/format.ts';

const MSE_COLOR = '#e0761f';
const BCE_COLOR = '#3b82f6';

function BinaryLossExplorer() {
  const [prediction, setPrediction] = useState(0.8);
  const [target, setTarget] = useState(1);
  const [lossName, setLossName] = useState<LossName>('bce');
  const [compare, setCompare] = useState(true);

  const loss = LOSSES[lossName];
  const value = loss.value([prediction], [target]);
  const gradient = loss.gradOutput([prediction], [target])[0];

  const yMax = lossName === 'bce' ? 4.2 : 1.1;

  return (
    <div className="grid grid--side">
      <Panel
        title="Loss as a function of the prediction"
        hint="drag the plot or the slider"
        caption="The curve is the loss for the current target across all possible predictions. The marker is the current prediction. When the two losses are overlaid, note how flat MSE becomes when the prediction is confidently wrong."
      >
        <Plot
          xDomain={[0.001, 0.999]}
          yDomain={[0, yMax]}
          height={300}
          xLabel="prediction ŷ"
          yLabel="loss L"
          onPointerData={(point) => setPrediction(Math.min(0.995, Math.max(0.005, point.x)))}
          ariaLabel="Loss curve"
        >
          {(scales) => (
            <>
              {compare && lossName === 'bce' ? (
                <Curve
                  f={(p) => LOSSES.mse.value([p], [target])}
                  scales={scales}
                  color={MSE_COLOR}
                  width={1.6}
                  opacity={0.7}
                  dash="5 4"
                />
              ) : null}
              {compare && lossName === 'mse' ? (
                <Curve
                  f={(p) => Math.min(yMax, LOSSES.bce.value([p], [target]))}
                  scales={scales}
                  color={BCE_COLOR}
                  width={1.6}
                  opacity={0.7}
                  dash="5 4"
                />
              ) : null}
              <Curve
                f={(p) => loss.value([p], [target])}
                scales={scales}
                color={lossName === 'bce' ? BCE_COLOR : MSE_COLOR}
                width={2.4}
              />
              <Marker
                x={prediction}
                y={Math.min(yMax, value)}
                scales={scales}
                color={lossName === 'bce' ? BCE_COLOR : MSE_COLOR}
                label={fmt(value, 3)}
              />
              <line
                x1={scales.x(target)}
                x2={scales.x(target)}
                y1={0}
                y2={scales.innerHeight}
                stroke="var(--text-faint)"
                strokeDasharray="3 3"
              />
              <text x={scales.x(target) + (target === 1 ? -6 : 6)} y={12} textAnchor={target === 1 ? 'end' : 'start'}>
                target y = {target}
              </text>
            </>
          )}
        </Plot>
      </Panel>

      <div className="stack">
        <Panel title="Controls">
          <div className="stack stack--sm">
            <Segmented
              label="Loss"
              value={lossName}
              options={[
                { value: 'bce' as LossName, label: 'Binary cross-entropy' },
                { value: 'mse' as LossName, label: 'Mean squared error' },
              ]}
              onChange={setLossName}
            />
            <Segmented
              label="Target"
              value={String(target)}
              options={[
                { value: '1', label: 'y = 1' },
                { value: '0', label: 'y = 0' },
              ]}
              onChange={(v) => setTarget(Number(v))}
            />
            <Slider
              label="Prediction ŷ"
              min={0.005}
              max={0.995}
              step={0.005}
              value={prediction}
              onChange={setPrediction}
              display={fmt(prediction, 3)}
            />
            <div className="btn-row">
              <Button onClick={() => setCompare(!compare)}>
                {compare ? 'Hide the other loss' : 'Overlay the other loss'}
              </Button>
            </div>
            {compare ? (
              <Legend
                items={[
                  { color: BCE_COLOR, label: 'binary cross-entropy' },
                  { color: MSE_COLOR, label: 'mean squared error' },
                ]}
              />
            ) : null}
          </div>
        </Panel>

        <Panel title="The calculation">
          <div className="calc">
            <div className="calc__line">
              <span className="calc__label">Target:     </span>
              {fmt(target, 1)}
            </div>
            <div className="calc__line">
              <span className="calc__label">Prediction: </span>
              {fmt(prediction, 3)}
            </div>
            {lossName === 'bce' ? (
              <>
                <div className="calc__line">
                  <span className="calc__label">L = </span>−[y·ln(ŷ) + (1−y)·ln(1−ŷ)]
                </div>
                <div className="calc__line">
                  <span className="calc__label">L = </span>
                  {target === 1
                    ? `−ln(${fmt(prediction, 3)})`
                    : `−ln(1 − ${fmt(prediction, 3)}) = −ln(${fmt(1 - prediction, 3)})`}
                </div>
              </>
            ) : (
              <>
                <div className="calc__line">
                  <span className="calc__label">L = </span>(ŷ − y)²
                </div>
                <div className="calc__line">
                  <span className="calc__label">L = </span>({fmt(prediction, 3)} − {fmt(target, 1)})² = (
                  {fmt(prediction - target, 3)})²
                </div>
              </>
            )}
            <div className="calc__line">
              <span className="calc__label">Loss:       </span>
              <span className="calc__result">{fmt(value, 4)}</span>
            </div>
            <div className="calc__line">
              <span className="calc__label">∂L/∂ŷ:      </span>
              {fmt(gradient, 4)}
            </div>
          </div>
        </Panel>

        <Stats
          items={[
            { label: 'Loss', value: fmt(value, 4), accent: true },
            { label: '∂L/∂ŷ', value: fmt(gradient, 3) },
            { label: 'Error ŷ − y', value: fmt(prediction - target, 3) },
          ]}
        />
      </div>
    </div>
  );
}

function CategoricalLossExplorer() {
  const [logits, setLogits] = useState([1.8, 0.4, -0.6]);
  const [correct, setCorrect] = useState(0);
  const probabilities = softmax(logits);
  const target = [0, 1, 2].map((i) => (i === correct ? 1 : 0));
  const value = LOSSES.cce.value(probabilities, target);
  const dz = probabilities.map((p, i) => p - target[i]);

  return (
    <div className="grid grid--side">
      <Panel
        title="Categorical cross-entropy over 3 classes"
        hint="only the correct class contributes"
        caption="Bars: predicted probabilities. The outlined bar is the true class. The loss is −log of that one probability; the other entries of a one-hot target multiply their logs by zero."
      >
        <svg className="viz" viewBox="0 0 480 200" role="img" aria-label="Categorical cross-entropy">
          {probabilities.map((p, i) => {
            const x = 60 + i * 130;
            const h = p * 120;
            const isTarget = i === correct;
            return (
              <g key={i}>
                <rect
                  x={x}
                  y={145 - h}
                  width={82}
                  height={Math.max(1, h)}
                  rx={3}
                  fill={isTarget ? '#3b82f6' : 'var(--text-faint)'}
                  fillOpacity={isTarget ? 0.45 : 0.2}
                  stroke={isTarget ? '#3b82f6' : 'var(--border-strong)'}
                  strokeWidth={isTarget ? 2 : 1}
                  style={{ transition: 'all 0.18s var(--ease)' }}
                />
                <text x={x + 41} y={145 - h - 8} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 12, fontWeight: 600 }}>
                  {fmt(p, 3)}
                </text>
                <text x={x + 41} y={161} textAnchor="middle">
                  class {i + 1}
                </text>
                <text x={x + 41} y={176} textAnchor="middle" style={{ fill: isTarget ? 'var(--accent-text)' : 'var(--text-faint)' }}>
                  y = {target[i]}
                </text>
                <text x={x + 41} y={191} textAnchor="middle" style={{ fill: 'var(--text-muted)' }}>
                  ∂L/∂z = {fmt(dz[i], 2)}
                </text>
              </g>
            );
          })}
          <line x1={40} y1={145} x2={460} y2={145} className="axis-line" />
          <line x1={40} y1={25} x2={460} y2={25} stroke="var(--border-strong)" strokeDasharray="4 4" />
          <text x={34} y={29} textAnchor="end">
            1.0
          </text>
        </svg>
      </Panel>

      <div className="stack">
        <Panel title="Logits and target">
          <div className="stack stack--sm">
            {logits.map((z, i) => (
              <Slider
                key={i}
                label={`z${i + 1}`}
                min={-4}
                max={4}
                value={z}
                onChange={(v) => {
                  const next = logits.slice();
                  next[i] = v;
                  setLogits(next);
                }}
                display={fmt(z, 2)}
              />
            ))}
            <Segmented
              label="True class"
              value={String(correct)}
              options={[
                { value: '0', label: 'class 1' },
                { value: '1', label: 'class 2' },
                { value: '2', label: 'class 3' },
              ]}
              onChange={(v) => setCorrect(Number(v))}
            />
          </div>
        </Panel>

        <Panel title="The calculation">
          <div className="calc">
            <div className="calc__line">
              <span className="calc__label">p       </span>= [{probabilities.map((p) => fmt(p, 3)).join(', ')}]
            </div>
            <div className="calc__line">
              <span className="calc__label">y       </span>= [{target.join(', ')}]
            </div>
            <div className="calc__line">
              <span className="calc__label">L = −Σ </span>yₖ·ln(pₖ) = −ln({fmt(probabilities[correct], 4)})
            </div>
            <div className="calc__line">
              <span className="calc__label">L       </span>= <span className="calc__result">{fmt(value, 4)}</span>
            </div>
            <div className="calc__line">
              <span className="calc__label">∂L/∂z   </span>= p − y = [{dz.map((d) => fmt(d, 3)).join(', ')}]
            </div>
          </div>
        </Panel>

        <Stats
          items={[
            { label: 'p of true class', value: fmt(probabilities[correct], 4) },
            { label: 'Loss', value: fmt(value, 4), accent: true },
            { label: 'Perplexity eᴸ', value: fmt(Math.exp(value), 3) },
          ]}
        />
      </div>
    </div>
  );
}

export function LossSection({ id, index }: SectionProps) {
  const [tab, setTab] = useState<'binary' | 'categorical'>('binary');
  const referenceTable = useMemo(
    () => [
      { y: 1, p: 0.99 },
      { y: 1, p: 0.9 },
      { y: 1, p: 0.8 },
      { y: 1, p: 0.5 },
      { y: 1, p: 0.2 },
      { y: 1, p: 0.01 },
    ],
    [],
  );

  return (
    <Section
      id={id}
      index={index}
      title="Loss functions"
      lede={
        <>
          A loss function reduces a prediction and its target to a single number that measures how
          wrong the prediction is. Training minimises the average of that number over the dataset.
          The choice of loss determines the gradient the network receives.
        </>
      }
    >
      <Equation caption="The objective: the mean loss over m training examples, as a function of the parameters θ = {W, b}.">
        {'J(\\theta) = \\frac{1}{m}\\sum_{i=1}^{m} L\\!\\left(\\hat{y}^{(i)}, y^{(i)}\\right)'}
      </Equation>

      <Segmented
        value={tab}
        options={[
          { value: 'binary' as const, label: 'Binary: MSE and cross-entropy' },
          { value: 'categorical' as const, label: 'Categorical cross-entropy' },
        ]}
        onChange={setTab}
        ariaLabel="Loss family"
      />

      {tab === 'binary' ? <BinaryLossExplorer /> : <CategoricalLossExplorer />}

      <div className="grid grid--side">
        <div className="prose-block">
          <h3 className="subhead">Why classification uses cross-entropy</h3>
          <p>
            Take <M>{'y = 1'}</M> and a sigmoid output. Under MSE the gradient with respect to the
            pre-activation is
          </p>
          <Equation plain>
            {"\\frac{\\partial L}{\\partial z} = 2(\\hat{y} - y)\\,\\sigma'(z) = 2(\\hat{y}-y)\\,\\hat{y}(1-\\hat{y})"}
          </Equation>
          <p>
            At <M>{'\\hat{y} = 0.01'}</M> — a confident, wrong prediction — that factor{' '}
            <M>{'\\hat{y}(1-\\hat{y}) \\approx 0.0099'}</M> nearly cancels the error, giving a
            gradient of about <M>{'-0.0196'}</M>. Under binary cross-entropy the same derivative is
          </p>
          <Equation plain>{'\\frac{\\partial L}{\\partial z} = \\hat{y} - y = -0.99'}</Equation>
          <p>
            The saturating factor cancels exactly against the <M>{'1/(\\hat{y}(1-\\hat{y}))'}</M> in
            the cross-entropy derivative. The gradient is then proportional to the error itself:
            large when the prediction is badly wrong, small when it is nearly right. That is the
            behaviour training needs.
          </p>
        </div>

        <Panel title="Cross-entropy for y = 1" flush>
          <table className="data">
            <thead>
              <tr>
                <th>ŷ</th>
                <th>BCE</th>
                <th>MSE</th>
                <th>∂L/∂z (BCE)</th>
                <th>∂L/∂z (MSE)</th>
              </tr>
            </thead>
            <tbody>
              {referenceTable.map((row) => {
                const bce = LOSSES.bce.value([row.p], [row.y]);
                const mse = LOSSES.mse.value([row.p], [row.y]);
                const dzBce = row.p - row.y;
                const dzMse = 2 * (row.p - row.y) * row.p * (1 - row.p);
                return (
                  <tr key={row.p}>
                    <td>{row.p.toFixed(2)}</td>
                    <td>{fmt(bce, 3)}</td>
                    <td>{fmt(mse, 3)}</td>
                    <td>{fmt(dzBce, 4)}</td>
                    <td>{fmt(dzMse, 4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      </div>

      <Note title="Units and interpretation" accent>
        <p>
          Cross-entropy measured with natural logarithms is in nats. A loss of{' '}
          <M>{'\\ln 2 \\approx 0.693'}</M> on a balanced binary problem means the model is no better
          than a coin flip; <M>{'\\ln K'}</M> is the equivalent baseline for <M>{'K'}</M> classes.
          Reporting <M>{'e^{L}'}</M> gives the effective number of classes the model is still
          undecided between.
        </p>
      </Note>
    </Section>
  );
}
