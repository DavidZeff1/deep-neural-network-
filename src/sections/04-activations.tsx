import { useMemo, useState } from 'react';
import { Section } from '../components/layout/Section.tsx';
import type { SectionProps } from './registry.ts';
import { ACTIVATIONS, softmax } from '../lib/activations.ts';
import type { ActivationName } from '../lib/activations.ts';
import { Panel, Note, Stats, Legend } from '../components/ui/layout.tsx';
import { Detail, InWords } from '../components/ui/Detail.tsx';
import { Segmented, Slider, Button } from '../components/ui/controls.tsx';
import { Equation, M } from '../components/ui/Math.tsx';
import { Curve, Marker, Plot } from '../components/viz/Plot.tsx';
import { fmt } from '../lib/format.ts';

const ORDER: ActivationName[] = ['relu', 'sigmoid', 'tanh', 'leakyRelu'];

const CURVE_COLORS: Record<ActivationName, string> = {
  relu: '#3b82f6',
  sigmoid: '#e0761f',
  tanh: '#16a34a',
  leakyRelu: '#8b5cf6',
  linear: '#64748b',
};

function SoftmaxPanel() {
  const [logits, setLogits] = useState([2.0, 1.0, 0.1]);
  const [temperature, setTemperature] = useState(1);
  const scaled = logits.map((z) => z / temperature);
  const probabilities = softmax(scaled);
  const max = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const entropy = -probabilities.reduce((acc, p) => acc + p * Math.log(Math.max(1e-12, p)), 0);
  const width = 480;
  const height = 190;
  const barWidth = 74;

  const setLogit = (index: number, value: number) => {
    const next = logits.slice();
    next[index] = value;
    setLogits(next);
  };

  return (
    <div className="grid grid--side">
      <Panel
        title="Softmax over 3 logits"
        hint="probabilities sum to 1"
        caption="Bars show the output probabilities. The dashed line marks the largest logit; softmax is invariant to adding a constant to every logit, which is why implementations subtract the maximum before exponentiating."
      >
        <svg className="viz" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Softmax probabilities">
          {probabilities.map((p, i) => {
            const x = 62 + i * (barWidth + 46);
            const h = p * 118;
            return (
              <g key={i}>
                <rect
                  x={x}
                  y={140 - h}
                  width={barWidth}
                  height={Math.max(1, h)}
                  rx={3}
                  fill="#3b82f6"
                  fillOpacity={0.28 + p * 0.55}
                  stroke="#3b82f6"
                  style={{ transition: 'all 0.18s var(--ease)' }}
                />
                <text x={x + barWidth / 2} y={140 - h - 8} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 12, fontWeight: 600 }}>
                  {fmt(p, 3)}
                </text>
                <text x={x + barWidth / 2} y={156} textAnchor="middle">
                  class {i + 1}
                </text>
                <text x={x + barWidth / 2} y={170} textAnchor="middle" style={{ fill: 'var(--text-muted)' }}>
                  z = {fmt(logits[i], 2)}
                </text>
              </g>
            );
          })}
          <line x1={40} y1={140} x2={width - 20} y2={140} className="axis-line" />
          <line x1={40} y1={22} x2={width - 20} y2={22} stroke="var(--border-strong)" strokeDasharray="4 4" />
          <text x={34} y={26} textAnchor="end">
            1.0
          </text>
          <text x={34} y={144} textAnchor="end">
            0
          </text>
        </svg>
      </Panel>

      <div className="stack">
        <Panel title="Logits">
          <div className="stack stack--sm">
            {logits.map((value, i) => (
              <Slider
                key={i}
                label={`z${i + 1}`}
                min={-4}
                max={4}
                value={value}
                onChange={(next) => setLogit(i, next)}
                display={fmt(value, 2)}
              />
            ))}
          </div>
          <div style={{ marginTop: 12 }}>
            <Slider
              label={<>Temperature T</>}
              min={0.1}
              max={4}
              step={0.05}
              value={temperature}
              onChange={setTemperature}
              display={fmt(temperature, 2)}
            />
          </div>
          <div className="btn-row" style={{ marginTop: 12 }}>
            <Button
              onClick={() => {
                setLogits([2, 1, 0.1]);
                setTemperature(1);
              }}
            >
              Reset
            </Button>
            <Button onClick={() => setLogits([0, 0, 0])}>All equal</Button>
            <Button onClick={() => setLogits([6, 1, 0.1])}>One dominant</Button>
            <Button onClick={() => setLogits([102, 101, 100.1])}>Large logits</Button>
          </div>
        </Panel>

        <Panel title="The calculation">
          <div className="calc">
            <div className="calc__line">
              <span className="calc__label">shift: </span>
              zᵢ − max(z) = {logits.map((z) => fmt(z - max, 2)).join(', ')}
            </div>
            <div className="calc__line">
              <span className="calc__label">exp:   </span>
              {exps.map((e) => fmt(e, 4)).join(', ')}
            </div>
            <div className="calc__line">
              <span className="calc__label">sum:   </span>
              {fmt(sum, 4)}
            </div>
            <div className="calc__line">
              <span className="calc__label">p:     </span>
              <span className="calc__result">{probabilities.map((p) => fmt(p, 4)).join(', ')}</span>
            </div>
            <div className="calc__line">
              <span className="calc__label">Σp:    </span>
              {fmt(probabilities.reduce((a, b) => a + b, 0), 6)}
            </div>
            <div className="calc__line">
              <span className="calc__label">H(p):  </span>
              {fmt(entropy, 4)} nats (max {fmt(Math.log(3), 4)})
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}

export function ActivationsSection({ id, index }: SectionProps) {
  const [name, setName] = useState<ActivationName>('relu');
  const [z, setZ] = useState(0.85);
  const [showAll, setShowAll] = useState(false);

  const act = ACTIVATIONS[name];
  const value = act.f(z);
  const slope = act.df(z);

  const yDomain = useMemo<[number, number]>(() => {
    if (name === 'relu' || name === 'leakyRelu') return [-1.2, 4.2];
    if (name === 'sigmoid') return [-0.25, 1.25];
    return [-1.35, 1.35];
  }, [name]);

  return (
    <Section
      id={id}
      index={index}
      title="Activation functions"
      lede={
        <>
          The activation <M>{'f'}</M> is applied elementwise to the pre-activations. Without it the
          whole network reduces to one linear map, so <M>{'f'}</M> is what makes depth meaningful.
          Its derivative decides how much gradient survives the trip back through the layer.
        </>
      }
    >
      <div className="grid grid--side">
        <Panel
          title="f(z) and its derivative"
          hint="drag the slider or the plot"
          caption="Solid line: f(z). Dashed line: f′(z). The marker is the current input. Flat regions of f are exactly the regions where f′ ≈ 0."
        >
          <Plot
            xDomain={[-4, 4]}
            yDomain={yDomain}
            height={300}
            xLabel="z"
            yLabel="f(z)"
            onPointerData={(point) => setZ(Math.max(-4, Math.min(4, point.x)))}
            ariaLabel={`${act.label} activation function`}
          >
            {(scales) => (
              <>
                {showAll
                  ? ORDER.filter((other) => other !== name).map((other) => (
                      <Curve
                        key={other}
                        f={ACTIVATIONS[other].f}
                        scales={scales}
                        color={CURVE_COLORS[other]}
                        width={1.2}
                        opacity={0.45}
                      />
                    ))
                  : null}
                <Curve
                  f={act.df}
                  scales={scales}
                  color={CURVE_COLORS[name]}
                  dash="4 4"
                  width={1.6}
                  opacity={0.75}
                  breaks={name === 'relu' || name === 'leakyRelu' ? [0] : undefined}
                />
                <Curve f={act.f} scales={scales} color={CURVE_COLORS[name]} width={2.4} />
                <Marker x={z} y={value} scales={scales} color={CURVE_COLORS[name]} />
                <Marker x={z} y={slope} scales={scales} color="var(--text-faint)" radius={3.4} guides={false} />
              </>
            )}
          </Plot>
        </Panel>

        <div className="stack">
          <Panel title="Function">
            <Segmented
              value={name}
              options={ORDER.map((n) => ({ value: n, label: ACTIVATIONS[n].label }))}
              onChange={setName}
              ariaLabel="Activation function"
            />
            <div style={{ marginTop: 16 }}>
              <Equation plain>{act.formula}</Equation>
              <Equation plain>{act.derivative}</Equation>
            </div>
            <div style={{ marginTop: 10 }}>
              <Slider
                label="Input z"
                min={-4}
                max={4}
                step={0.01}
                value={z}
                onChange={setZ}
                display={fmt(z, 2)}
              />
            </div>
            <div className="btn-row" style={{ marginTop: 10 }}>
              <Button onClick={() => setShowAll(!showAll)}>
                {showAll ? 'Hide other functions' : 'Overlay other functions'}
              </Button>
            </div>
            {showAll ? (
              <div style={{ marginTop: 10 }}>
                <Legend
                  items={ORDER.map((n) => ({ color: CURVE_COLORS[n], label: ACTIVATIONS[n].label }))}
                />
              </div>
            ) : null}
          </Panel>

          <Stats
            items={[
              { label: 'input z', value: fmt(z, 3) },
              { label: 'output f(z)', value: fmt(value, 3), accent: true },
              { label: "slope f'(z)", value: fmt(slope, 3) },
            ]}
          />

          <Note>
            <p style={{ fontSize: 14 }}>{act.notes}</p>
          </Note>
        </div>
      </div>

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">What f has to provide</h3>
          <p>
            Only two properties are essential. <M>{'f'}</M> must be non-linear, or the whole network
            collapses to one affine map. And it must be differentiable almost everywhere, or
            gradient descent has nothing to work with. Everything else — the exact shape, the
            output range, whether it saturates — is a trade-off rather than a requirement.
          </p>
          <p>
            The four functions here differ in three measurable ways: the range of{' '}
            <M>{'f'}</M>, the maximum of <M>{"f'"}</M>, and whether <M>{"f'"}</M> reaches zero.
          </p>
          <table className="data" style={{ marginBottom: 14 }}>
            <thead>
              <tr>
                <th>f</th>
                <th>range</th>
                <th>max f′</th>
                <th>f′ → 0</th>
                <th>zero-centred</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>ReLU</td>
                <td>[0, ∞)</td>
                <td>1</td>
                <td>z &lt; 0</td>
                <td>no</td>
              </tr>
              <tr>
                <td>Leaky ReLU</td>
                <td>(−∞, ∞)</td>
                <td>1</td>
                <td>never</td>
                <td>no</td>
              </tr>
              <tr>
                <td>Sigmoid</td>
                <td>(0, 1)</td>
                <td>0.25</td>
                <td>|z| large</td>
                <td>no</td>
              </tr>
              <tr>
                <td>Tanh</td>
                <td>(−1, 1)</td>
                <td>1</td>
                <td>|z| large</td>
                <td>yes</td>
              </tr>
            </tbody>
          </table>
          <p>
            The <em>max f′</em> column is the one that decides whether a deep stack trains, and the{' '}
            <em>f′ → 0</em> column decides whether individual units can stop learning permanently.
          </p>

          <h3 className="subhead">Why the derivative is the quantity that matters</h3>
          <p>
            During backpropagation the gradient arriving at a unit is multiplied by{' '}
            <M>{"f'(z)"}</M> before it continues to the previous layer. Across{' '}
            <M>{'L'}</M> layers the factors multiply:
          </p>
          <Equation>
            {"\\frac{\\partial L}{\\partial \\mathbf{a}^{(0)}} \\propto \\prod_{l=1}^{L} f'\\!\\left(z^{(l)}\\right) \\cdot W^{(l)}"}
          </Equation>
          <InWords>
            <p>
              <M>{'\\prod'}</M> means multiply the terms together, the way{' '}
              <M>{'\\sum'}</M> means add them. So this says: the influence an early layer has on
              the final error is the product of one factor per layer in between.
            </p>
            <p>
              That is the chain rule from section 00 applied <M>{'L'}</M> times. If each factor is a
              bit less than 1, multiplying many of them together drives the whole product towards
              zero — and then the early layers stop learning.
            </p>
          </InWords>
          <p>
            With sigmoid, <M>{"f'(z) \\le 0.25"}</M> everywhere. Ten such layers contribute a factor
            of at most <M>{'0.25^{10} \\approx 10^{-6}'}</M>, so the early layers receive almost no
            gradient. This is the vanishing-gradient problem, and it is the practical reason ReLU
            replaced sigmoid in hidden layers: for <M>{'z > 0'}</M> its derivative is exactly 1, so
            the product does not shrink.
          </p>

          <Detail kicker="worked example" title="Dead ReLU units">
            <p>
              A ReLU unit with <M>{'z < 0'}</M> for every training example outputs 0 for all of them
              and has <M>{"f'(z) = 0"}</M> for all of them. Its <M>{'\\delta'}</M> is therefore 0,
              so the gradients of its incoming weights and its bias are all 0, and no update
              changes them. The unit is permanently dead — not slow, dead.
            </p>
            <p>
              Concretely: take a unit with <M>{'\\mathbf{w} = (0.4, -0.2)'}</M> and{' '}
              <M>{'b = -3.0'}</M>, on data confined to{' '}
              <M>{'[-1,1]^2'}</M>. The largest achievable <M>{'z'}</M> is{' '}
              <M>{'0.4 + 0.2 - 3.0 = -2.4 < 0'}</M>, so the unit never fires. Its gradient is
              exactly zero on every example. Nothing in the training loop can revive it, because the
              only term that could move <M>{'b'}</M> is <M>{'\\delta = 0'}</M>.
            </p>
            <p>
              How units get there: a learning rate large enough to push the bias strongly negative
              in one step. This is a real failure mode — a fraction of units in a ReLU network
              typically ends up dead — and it is why leaky ReLU exists. With{' '}
              <M>{'\\alpha = 0.1'}</M> the same unit still has{' '}
              <M>{"f'(z) = 0.1"}</M>, so its gradient is a tenth of full strength rather than zero,
              and it can recover.
            </p>
          </Detail>

          <Detail title="tanh is a rescaled sigmoid">
            <p>Start from the definitions and multiply numerator and denominator by <M>{'e^{z}'}</M>:</p>
            <Equation plain>
              {'\\tanh(z) = \\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} = \\frac{e^{2z}-1}{e^{2z}+1}'}
            </Equation>
            <p>
              Now write <M>{'\\sigma(2z) = 1/(1+e^{-2z}) = e^{2z}/(e^{2z}+1)'}</M> and compute:
            </p>
            <Equation plain>
              {'2\\sigma(2z) - 1 = \\frac{2e^{2z}}{e^{2z}+1} - \\frac{e^{2z}+1}{e^{2z}+1} = \\frac{e^{2z}-1}{e^{2z}+1} = \\tanh(z)'}
            </Equation>
            <p>
              So <M>{'\\tanh(z) = 2\\sigma(2z) - 1'}</M>. The two are the same function up to a
              rescaling of the input and an affine map of the output, which means a network can
              convert one into the other by adjusting weights and biases — the layer before can
              supply the factor of 2, the layer after can supply the shift and scale.
            </p>
            <p>
              They are nevertheless not interchangeable in practice. Differentiating the identity
              gives <M>{"\\tanh'(z) = 4\\sigma'(2z)"}</M>, so tanh's derivative peaks at 1 while
              sigmoid's peaks at 0.25. And tanh is zero-centred, so a layer's outputs have mean near
              zero rather than near 0.5, which keeps the next layer's pre-activations centred rather
              than drifting with every added layer.
            </p>
          </Detail>
        </div>

        <div className="prose-block">
          <h3 className="subhead">Choosing an output activation</h3>
          <ul>
            <li>
              <strong>Regression:</strong> linear. The output must be able to take any real value.
            </li>
            <li>
              <strong>Binary classification:</strong> sigmoid, giving one number in{' '}
              <M>{'(0,1)'}</M> read as <M>{'P(y = 1 \\mid \\mathbf{x})'}</M>.
            </li>
            <li>
              <strong>K mutually exclusive classes:</strong> softmax, giving <M>{'K'}</M> positive
              numbers that sum to 1.
            </li>
            <li>
              <strong>K independent labels:</strong> K separate sigmoids, not softmax — the labels
              are not competing for a shared budget of probability.
            </li>
          </ul>
          <p>
            Hidden layers are a separate choice; ReLU is the standard default, with tanh preferred
            in small networks where zero-centred activations help.
          </p>

          <h3 className="subhead">Logits</h3>
          <p>
            The pre-activations of the final layer are called logits. The name comes from the
            inverse of the sigmoid, the logit function{' '}
            <M>{'\\operatorname{logit}(p) = \\log\\frac{p}{1-p}'}</M>: applying it to a
            probability recovers the <M>{'z'}</M> that produced it. So a logit is a log-odds.
          </p>
          <p>
            A logit of 0 is probability 0.5, a logit of 2 is{' '}
            <M>{'\\sigma(2) = 0.881'}</M>, a logit of 5 is 0.993. The map is compressive: moving a
            logit from 5 to 6 changes the probability by 0.004, while moving it from 0 to 1 changes
            it by 0.231. This is exactly the saturation that makes the gradient small at large{' '}
            <M>{'|z|'}</M>, and the reason losses and metrics are usually computed from logits
            rather than from probabilities.
          </p>
          <p>
            Libraries keep the logits and fuse the activation into the loss for the same reason:{' '}
            <code>sigmoid_cross_entropy_with_logits</code> never materialises{' '}
            <M>{'\\hat{y}'}</M>, so it never has to divide by <M>{'\\hat{y}(1-\\hat{y})'}</M>{' '}
            and never loses precision when that quantity is near zero.
          </p>
        </div>
      </div>

      <h3 className="subhead">Softmax</h3>
      <p className="prose-block">
        Softmax maps a vector of <M>{'K'}</M> real numbers to a probability distribution. Unlike the
        functions above it is not elementwise: every output depends on every input.
      </p>
      <Equation caption="Exponentiation makes every entry positive; dividing by the sum makes them add to 1.">
        {'\\mathrm{softmax}(\\mathbf{z})_k = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j}}'}
      </Equation>
      <InWords>
        <p>
          Two steps. First raise <M>{'e \\approx 2.718'}</M> to the power of each number, which
          turns any value — including negative ones — into a positive one, and exaggerates the
          differences between them. Then divide each result by the total of all of them, which
          forces the final numbers to add up to exactly 1.
        </p>
        <p>
          Numbers that are positive and add to 1 can be read as probabilities, which is the whole
          point: the network's raw outputs become "how likely is each class".
        </p>
      </InWords>

      <SoftmaxPanel />

      <div className="grid grid--2">
        <div className="prose-block">
          <h3 className="subhead">Temperature</h3>
          <p>
            Dividing the logits by a constant <M>{'T'}</M> before the softmax rescales how sharply
            the distribution concentrates. The ordering of the classes never changes — division by a
            positive constant is monotonic — but the confidence does.
          </p>
          <Equation plain>
            {'p_k(T) = \\frac{e^{z_k/T}}{\\sum_j e^{z_j/T}}'}
          </Equation>
          <ul>
            <li>
              <M>{'T \\to 0'}</M>: the largest logit dominates completely and{' '}
              <M>{'p'}</M> approaches a one-hot vector at the argmax.
            </li>
            <li>
              <M>{'T = 1'}</M>: the ordinary softmax.
            </li>
            <li>
              <M>{'T \\to \\infty'}</M>: every exponent goes to 0, so{' '}
              <M>{'p'}</M> approaches the uniform distribution <M>{'1/K'}</M>.
            </li>
          </ul>
          <p>
            The entropy readout in the calculation panel quantifies this: it runs from 0 at{' '}
            <M>{'T \\to 0'}</M> up to <M>{'\\log K = 1.0986'}</M> nats for three classes at{' '}
            <M>{'T \\to \\infty'}</M>. Temperature is used at sampling time to control how
            deterministic a model's choices are; it is not usually a trained parameter.
          </p>
        </div>

        <div className="prose-block">
          <h3 className="subhead">Numerical stability</h3>
          <p>
            Press <strong>Large logits</strong> above. The logits become 102, 101, 100.1 — the same
            differences as the default, so the probabilities are identical. A naive implementation
            would evaluate <M>{'e^{102} \\approx 10^{44}'}</M>, which is finite in double
            precision but overflows to <M>{'\\infty'}</M> in 32-bit floats, and the division
            returns NaN.
          </p>
          <p>
            Subtracting the maximum first fixes this exactly, not approximately, because of the
            shift invariance below: the largest shifted exponent is <M>{'e^{0} = 1'}</M> and every
            other is in <M>{'(0, 1]'}</M>. The calculation panel shows the shifted values.
          </p>
          <p>
            The same problem appears in the loss. Computing <M>{'\\log p_k'}</M> by first forming{' '}
            <M>{'p_k'}</M> can underflow to <M>{'\\log 0 = -\\infty'}</M>. The log-sum-exp form
            avoids it:
          </p>
          <Equation plain>
            {'\\log p_k = z_k - m - \\log\\!\\sum_j e^{z_j - m}, \\quad m = \\max_j z_j'}
          </Equation>
          <p>
            Every term is finite here, which is why frameworks expose{' '}
            <code>log_softmax</code> as a primitive rather than composing a log with a softmax.
          </p>
        </div>
      </div>

      <Note title="Two properties worth knowing" accent>
        <p>
          <strong>Shift invariance:</strong> adding a constant <M>{'c'}</M> to every logit leaves
          the output unchanged, because <M>{'e^{z_k + c} = e^{c} e^{z_k}'}</M> cancels between
          numerator and denominator. Set all three sliders to the same value to see it. One
          consequence: softmax has <M>{'K'}</M> outputs but only <M>{'K-1'}</M> degrees of freedom,
          so the final layer's weights are not identified — adding a constant vector to every row of{' '}
          <M>{'W^{(L)}'}</M> leaves the network's predictions unchanged.
        </p>
        <p>
          <strong>Jacobian:</strong> since each output depends on all inputs, the derivative is a
          matrix, <M>{'\\partial p_i / \\partial z_j = p_i(\\delta_{ij} - p_j)'}</M>. Composed with
          categorical cross-entropy it collapses to <M>{'\\partial L/\\partial z_k = p_k - y_k'}</M>,
          which is why the two are almost always implemented together. Section 08 carries out that
          cancellation.
        </p>
      </Note>
    </Section>
  );
}
