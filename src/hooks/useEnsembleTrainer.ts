import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MLP } from '../lib/network.ts';
import type { NetworkConfig, Sample } from '../lib/network.ts';

export interface EnsembleMember {
  label: string;
  config: NetworkConfig;
}

export interface EnsembleResult {
  label: string;
  network: MLP;
  loss: number;
  accuracy: number;
  history: Array<[number, number]>;
  parameterCount: number;
}

export interface EnsembleTrainer {
  results: EnsembleResult[];
  epoch: number;
  running: boolean;
  version: number;
  start: () => void;
  pause: () => void;
  reset: () => void;
  runEpochs: (count: number) => void;
}

/**
 * Trains several networks in lockstep on the same data so their behaviour can
 * be compared at equal epoch counts.
 */
export function useEnsembleTrainer(
  members: EnsembleMember[],
  train: Sample[],
  learningRate: number,
  batchSize: number,
  epochsPerTick = 2,
  maxEpochs = 2000,
): EnsembleTrainer {
  const key = useMemo(
    () => JSON.stringify(members.map((m) => m.config)) + `|${train.length}|${train[0]?.x.join(',') ?? ''}`,
    [members, train],
  );

  const netsRef = useRef<MLP[]>(members.map((m) => new MLP(m.config)));
  const historyRef = useRef<Array<Array<[number, number]>>>(members.map(() => []));
  const epochRef = useRef(0);
  const runningRef = useRef(false);
  const frameRef = useRef<number | null>(null);

  const [running, setRunning] = useState(false);
  const [version, setVersion] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [snapshot, setSnapshot] = useState<Array<{ loss: number; accuracy: number }>>(
    members.map(() => ({ loss: 0, accuracy: 0 })),
  );

  const hyper = useRef({ learningRate, batchSize, epochsPerTick, maxEpochs });
  hyper.current = { learningRate, batchSize, epochsPerTick, maxEpochs };
  const dataRef = useRef(train);
  dataRef.current = train;
  const membersRef = useRef(members);
  membersRef.current = members;

  const measure = useCallback(
    () =>
      netsRef.current.map((net) => ({
        loss: net.evaluateLoss(dataRef.current),
        accuracy: net.evaluateAccuracy(dataRef.current),
      })),
    [],
  );

  const reset = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    netsRef.current = membersRef.current.map((m) => new MLP(m.config));
    epochRef.current = 0;
    const measured = measure();
    historyRef.current = measured.map((m) => [[0, m.loss] as [number, number]]);
    setSnapshot(measured);
    setEpoch(0);
    setVersion((v) => v + 1);
  }, [measure]);

  useEffect(() => {
    reset();
    // Keyed on the serialised member configs and dataset only.
  }, [key]);

  const runOne = useCallback(() => {
    for (const net of netsRef.current) {
      net.trainEpoch(dataRef.current, hyper.current.learningRate, hyper.current.batchSize);
    }
    epochRef.current += 1;
    const measured = measure();
    measured.forEach((m, i) => historyRef.current[i].push([epochRef.current, m.loss]));
    return measured;
  }, [measure]);

  const flush = useCallback((measured: Array<{ loss: number; accuracy: number }>) => {
    setSnapshot(measured);
    setEpoch(epochRef.current);
    setVersion((v) => v + 1);
  }, []);

  const tick = useCallback(() => {
    if (!runningRef.current) return;
    let measured = measure();
    for (let i = 0; i < hyper.current.epochsPerTick; i++) {
      if (epochRef.current >= hyper.current.maxEpochs) {
        runningRef.current = false;
        setRunning(false);
        break;
      }
      measured = runOne();
    }
    flush(measured);
    if (runningRef.current) frameRef.current = requestAnimationFrame(tick);
  }, [runOne, flush, measure]);

  const start = useCallback(() => {
    if (runningRef.current) return;
    runningRef.current = true;
    setRunning(true);
    frameRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const pause = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
  }, []);

  const runEpochs = useCallback(
    (count: number) => {
      pause();
      let measured = measure();
      for (let i = 0; i < count; i++) measured = runOne();
      flush(measured);
    },
    [pause, runOne, flush, measure],
  );

  useEffect(
    () => () => {
      runningRef.current = false;
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    },
    [],
  );

  const results: EnsembleResult[] = members.map((member, i) => ({
    label: member.label,
    network: netsRef.current[i],
    loss: snapshot[i]?.loss ?? 0,
    accuracy: snapshot[i]?.accuracy ?? 0,
    history: historyRef.current[i] ?? [],
    parameterCount: netsRef.current[i]?.parameterCount() ?? 0,
  }));

  return { results, epoch, running, version, start, pause, reset, runEpochs };
}
