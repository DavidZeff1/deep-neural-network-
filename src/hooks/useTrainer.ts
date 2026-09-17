import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MLP } from '../lib/network.ts';
import type { NetworkConfig, Sample } from '../lib/network.ts';

export interface MetricPoint {
  epoch: number;
  trainLoss: number;
  validationLoss: number;
  trainAccuracy: number;
  validationAccuracy: number;
}

export interface TrainerOptions {
  /** Must be referentially stable between renders (memoise it). */
  config: NetworkConfig;
  train: Sample[];
  validation?: Sample[];
  learningRate: number;
  batchSize: number;
  /** Epochs executed per animation frame while running. */
  epochsPerTick?: number;
  /** Training stops automatically at this epoch. */
  maxEpochs?: number;
}

export interface Trainer {
  network: MLP;
  history: MetricPoint[];
  metrics: MetricPoint;
  running: boolean;
  /** Increments on every parameter change; use as a redraw key. */
  version: number;
  start: () => void;
  pause: () => void;
  reset: (seed?: number) => void;
  runEpochs: (count: number) => void;
  /** Restores the parameters recorded at the epoch with the lowest validation loss. */
  restoreBest: () => void;
  bestEpoch: MetricPoint | null;
}

const EMPTY: MetricPoint = {
  epoch: 0,
  trainLoss: 0,
  validationLoss: 0,
  trainAccuracy: 0,
  validationAccuracy: 0,
};

/**
 * Runs mini-batch gradient descent in the background, one or more epochs per
 * animation frame, and exposes the metric history for plotting.
 *
 * Hyperparameters are read from refs on every tick, so changing the learning
 * rate or batch size mid-run takes effect immediately without restarting.
 */
export function useTrainer(options: TrainerOptions): Trainer {
  const {
    config,
    train,
    validation,
    learningRate,
    batchSize,
    epochsPerTick = 3,
    maxEpochs = 4000,
  } = options;

  // L2 and dropout are applied live, so they must not trigger a rebuild.
  const configKey = useMemo(() => {
    const { l2: _l2, dropout: _dropout, ...architecture } = config;
    return JSON.stringify(architecture);
  }, [config]);
  const dataKey = useMemo(
    () => `${train.length}:${train[0]?.x.join(',') ?? ''}:${validation?.length ?? 0}:${validation?.[0]?.x.join(',') ?? ''}`,
    [train, validation],
  );

  const networkRef = useRef<MLP>(new MLP(config));
  const historyRef = useRef<MetricPoint[]>([]);
  const bestRef = useRef<{ metric: MetricPoint; state: { W: number[][][]; b: number[][] } } | null>(
    null,
  );
  const runningRef = useRef(false);
  const frameRef = useRef<number | null>(null);

  const [running, setRunning] = useState(false);
  const [version, setVersion] = useState(0);
  const [history, setHistory] = useState<MetricPoint[]>([]);
  const [metrics, setMetrics] = useState<MetricPoint>(EMPTY);

  const hyper = useRef({ learningRate, batchSize, epochsPerTick, maxEpochs });
  hyper.current = { learningRate, batchSize, epochsPerTick, maxEpochs };

  const data = useRef({ train, validation });
  data.current = { train, validation };

  const measure = useCallback((): MetricPoint => {
    const net = networkRef.current;
    const { train: trainSet, validation: validationSet } = data.current;
    return {
      epoch: historyRef.current.length,
      trainLoss: net.evaluateLoss(trainSet),
      validationLoss: validationSet && validationSet.length > 0 ? net.evaluateLoss(validationSet) : 0,
      trainAccuracy: net.evaluateAccuracy(trainSet),
      validationAccuracy:
        validationSet && validationSet.length > 0 ? net.evaluateAccuracy(validationSet) : 0,
    };
  }, []);

  const hardReset = useCallback(
    (seed?: number) => {
      runningRef.current = false;
      setRunning(false);
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
      networkRef.current = new MLP(seed === undefined ? config : { ...config, seed });
      historyRef.current = [];
      const initial = measure();
      historyRef.current = [initial];
      // Epoch 0 is a legitimate "best so far", so early stopping is available
      // from the start rather than only after the first epoch.
      bestRef.current = { metric: initial, state: networkRef.current.snapshot() };
      setHistory([initial]);
      setMetrics(initial);
      setVersion((v) => v + 1);
    },
    [config, measure],
  );

  useEffect(() => {
    networkRef.current.setRegularisation({ l2: config.l2 ?? 0, dropout: config.dropout ?? 0 });
  }, [config.l2, config.dropout, version]);

  // Rebuild whenever the architecture or the dataset changes.
  useEffect(() => {
    hardReset();
    // Keyed on the serialised architecture and dataset only: hardReset is
    // recreated on every config change and would otherwise loop.
  }, [configKey, dataKey]);

  const runOneEpoch = useCallback(() => {
    const net = networkRef.current;
    const { train: trainSet } = data.current;
    net.trainEpoch(trainSet, hyper.current.learningRate, hyper.current.batchSize);
    const point = measure();
    historyRef.current.push(point);
    const best = bestRef.current;
    const hasValidation = (data.current.validation?.length ?? 0) > 0;
    const score = hasValidation ? point.validationLoss : point.trainLoss;
    const bestScore = best ? (hasValidation ? best.metric.validationLoss : best.metric.trainLoss) : Infinity;
    if (!best || score < bestScore) {
      bestRef.current = { metric: point, state: net.snapshot() };
    }
    return point;
  }, [measure]);

  const flush = useCallback(() => {
    setHistory(historyRef.current.slice());
    setMetrics(historyRef.current[historyRef.current.length - 1] ?? EMPTY);
    setVersion((v) => v + 1);
  }, []);

  const tick = useCallback(() => {
    if (!runningRef.current) return;
    for (let i = 0; i < hyper.current.epochsPerTick; i++) {
      if (historyRef.current.length > hyper.current.maxEpochs) {
        runningRef.current = false;
        setRunning(false);
        break;
      }
      runOneEpoch();
    }
    flush();
    if (runningRef.current) frameRef.current = requestAnimationFrame(tick);
  }, [runOneEpoch, flush]);

  const start = useCallback(() => {
    if (runningRef.current) return;
    if (historyRef.current.length === 0) historyRef.current = [measure()];
    runningRef.current = true;
    setRunning(true);
    frameRef.current = requestAnimationFrame(tick);
  }, [tick, measure]);

  const pause = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
  }, []);

  const runEpochs = useCallback(
    (count: number) => {
      pause();
      if (historyRef.current.length === 0) historyRef.current = [measure()];
      for (let i = 0; i < count; i++) runOneEpoch();
      flush();
    },
    [pause, runOneEpoch, flush, measure],
  );

  const restoreBest = useCallback(() => {
    const best = bestRef.current;
    if (!best) return;
    pause();
    networkRef.current.restore(best.state);
    const point = measure();
    historyRef.current.push({ ...point, epoch: historyRef.current.length });
    flush();
  }, [pause, measure, flush]);

  useEffect(
    () => () => {
      runningRef.current = false;
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    },
    [],
  );

  return {
    network: networkRef.current,
    history,
    metrics,
    running,
    version,
    start,
    pause,
    reset: hardReset,
    runEpochs,
    restoreBest,
    bestEpoch: bestRef.current?.metric ?? null,
  };
}
