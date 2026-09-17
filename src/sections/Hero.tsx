import { useEffect, useMemo, useState } from 'react';
import { MLP } from '../lib/network.ts';
import { NetworkDiagram } from '../components/viz/NetworkDiagram.tsx';
import { Button } from '../components/ui/controls.tsx';

/** A small network with a slowly drifting input, purely to show the notation in use. */
function LiveDiagram() {
  const network = useMemo(
    () =>
      new MLP({
        inputSize: 3,
        hiddenUnits: [5, 4],
        outputSize: 2,
        hiddenActivation: 'tanh',
        outputActivation: 'softmax',
        loss: 'cce',
        seed: 1234,
      }),
    [],
  );
  const [t, setT] = useState(0);

  useEffect(() => {
    let frame = 0;
    let running = true;
    const tick = () => {
      if (!running) return;
      frame += 1;
      if (frame % 3 === 0) setT((value) => value + 0.03);
      requestAnimationFrame(tick);
    };
    const handle = requestAnimationFrame(tick);
    return () => {
      running = false;
      cancelAnimationFrame(handle);
    };
  }, []);

  const trace = useMemo(
    () => network.forward([Math.sin(t), Math.cos(t * 0.7), Math.sin(t * 1.3 + 1)]),
    [network, t],
  );

  return (
    <NetworkDiagram
      network={network}
      trace={trace}
      height={260}
      showValues
      inputLabels={['x₁', 'x₂', 'x₃']}
      outputLabels={['ŷ₁', 'ŷ₂']}
    />
  );
}

export function Hero({ onStart }: { onStart: () => void }) {
  return (
    <header className="hero" id="top">
      <h1 className="hero__title">Deep neural networks, one computation at a time</h1>
      <div className="grid grid--side" style={{ alignItems: 'start', gap: 32 }}>
        <div>
          <p className="hero__lede">
            A neural network is a chain of matrix multiplications with a non-linear function applied
            after each one. Everything else — training, backpropagation, regularisation — follows
            from that definition and from calculus.
          </p>
          <p>
            Every figure below is live. The networks are real: weights are initialised, forward
            passes are computed, gradients are derived by backpropagation and verified against
            finite differences, and training runs in your browser. Change a parameter and the
            numbers next to it change with it.
          </p>
          <div className="btn-row" style={{ marginTop: 20 }}>
            <Button variant="primary" onClick={onStart}>
              Start with network structure
            </Button>
          </div>
          <div className="row" style={{ marginTop: 22, gap: 18 }}>
            <span className="tag">12 sections</span>
            <span className="tag">no pre-trained weights</span>
            <span className="tag">runs offline</span>
          </div>
        </div>
        <div className="panel">
          <div className="panel__head">
            <span className="panel__title">3 → 5 → 4 → 2, tanh hidden, softmax output</span>
          </div>
          <div className="panel__body" style={{ padding: 8 }}>
            <LiveDiagram />
          </div>
          <div className="figure__caption">
            Node shading shows each activation; edge width and colour show each weight. Blue is
            positive, orange negative.
          </div>
        </div>
      </div>
    </header>
  );
}
