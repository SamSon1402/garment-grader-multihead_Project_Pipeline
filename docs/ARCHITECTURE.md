# Architecture notes

## 1. Safety boundary

The real-time line should continue to make a safe decision when cloud services are unavailable.

```text
camera/NIR → local validation → local inference → local policy → PLC/review
                                      │
                                      └→ local event queue → cloud when available
```

A network call is therefore not allowed between inference and the PLC command.

## 2. Camera and sensor health are separate from model health

A classifier can be healthy while the camera is overexposed. This repository runs cheap input checks before the model and records those failures separately.

Production checks should also include dropped frames, exposure/gain drift, color target calibration, trigger timing, NIR calibration and camera heartbeat.

## 3. Perception is not grading policy

Models produce evidence. Customer policy converts evidence into business decisions.

This prevents a common architecture mistake where every customer rule change becomes a model retraining task.

## 4. Confidence and OOD can override an apparently good grade

A high class probability is not enough. If the sample is out-of-distribution or any required sensor fails, the safe result is `REVIEW`.

## 5. TensorRT path

For a real trained model, the intended release pipeline is:

```text
PyTorch
  ↓
ONNX export
  ↓
ONNX Runtime numerical check
  ↓
TensorRT FP16
  ↓
benchmark on target Jetson
  ↓
INT8 only if needed
  ↓
accuracy/equivalence gate
  ↓
container + model version
  ↓
shadow
  ↓
canary
  ↓
production
```

The TensorRT provider is intentionally not fabricated without an exact model contract. A CTO should be able to see exactly where the real engine belongs and what must be tested around it.

## 6. DLP projection

Projection is not required for every garment. Its useful role is human review, operator guidance and debugging.

```text
camera zone -------------------------- projection zone
    capture                                   ↓
       ↓                                  GRADE B
    inference                              ROUTE 04
       ↓                                   STAIN ○
    decision
       ↓
encoder predicts garment arrival
```

Camera coordinates are transformed to projector coordinates using a calibrated homography. A real conveyor should use encoder pulses rather than wall-clock timing as the final source of truth.

## 7. Failure path

```text
bad RGB frame --------┐
bad NIR signal -------┤
OOD ------------------┤
low confidence -------┤
no policy match ------┘
          ↓
       REVIEW
          ↓
projection/operator station
          ↓
human label / correction
          ↓
future active-learning dataset
```

This makes failure visible instead of silently producing a confident wrong route.

## 8. Next project connection

The emitted `GarmentEvent` is intentionally rich enough to feed **SortDrift Sentinel** later:

- model version
- customer + policy version
- confidence
- OOD score
- defect distribution
- latency
- input quality
- grade and route

That lets the second project monitor camera drift, model drift, runtime health and business outcomes without tightly coupling it to this service.
