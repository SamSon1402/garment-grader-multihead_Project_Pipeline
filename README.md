# GarmentGrader Industrial

A small but production-shaped **computer vision system for grading and routing second-hand garments on a conveyor**.

It is intentionally not a full commercial product. It contains the core code a technical reviewer would expect to see: input validation, model-provider abstraction, multimodal RGB + NIR results, grading policy, routing, projection commands, event storage, metrics, tests, Docker, and clear failure paths.

## The problem

A textile sorting line has to answer several questions quickly:

- What garment is this?
- What brand / attributes does it have?
- Is there a stain, hole, tear, or other defect?
- What is the likely fiber composition from NIR?
- What grade should the garment receive?
- Where should it be routed?
- What happens when confidence is low or a camera is unhealthy?

The important part is not only the model. The whole decision has to be reliable and fast enough for a moving conveyor.

## What this project does

```text
GARMENT
   ↓
trigger + encoder
   ↓
RGB image ───────────────┐
                        │
NIR reading ─────────────┤
                        ↓
                 input validation
                        ↓
                 CV inference layer
                        ↓
       category / brand / attributes
       defects / confidence / embedding
       fiber composition
                        ↓
                 grading policy
                        ↓
                 pricing estimate
                        ↓
                  routing policy
                        ↓
        ┌───────────────┼────────────────┐
        ↓               ↓                ↓
       PLC         DLP projection      event
     command       grade/defect      telemetry
                                    + storage
```

The demo uses a deterministic local model provider so the repository runs without model weights or paid accounts. The interfaces are real and can be switched to Roboflow/self-hosted inference or a future TensorRT runtime.

## Why this architecture

The line keeps the **real-time decision path local**. Camera validation, inference, grading, and routing do not depend on cloud availability.

The cloud/control-plane integrations are outside the hot path. If Supabase or another backend is unavailable, the edge process can keep producing decisions and write events to a local JSONL queue.

The grading logic is also separated from perception. A model reports evidence. A customer policy decides what that evidence means for grade and route.

```text
PERCEPTION: "stain area = 1.4%, hole = 0 mm, confidence = 0.94"
                         ↓
CUSTOMER POLICY: "for this customer, that means Grade B"
                         ↓
ROUTING: "resale_lane_2"
```

That separation makes customer onboarding and rule changes much safer than retraining a model for every business rule.

## YC products used as building blocks

This repository uses or provides adapters for public products from YC companies. They are optional so the demo still runs locally.

- **Roboflow (YC S20)** — optional self-hosted/hosted computer-vision inference adapter. Roboflow provides dataset, model and deployment tooling, including self-hosted inference.  
  https://www.ycombinator.com/companies/roboflow  
  https://inference.roboflow.com/

- **Supabase (YC S20)** — optional Postgres event/control-plane sink. It is deliberately not required for the real-time edge decision.  
  https://www.ycombinator.com/companies/supabase

- **Modal (YC W21)** — optional offline GPU benchmark job. It belongs in experimentation / benchmarking, not the conveyor hot path.  
  https://www.ycombinator.com/companies/modal

## YC companies used as architecture references

These are **references, not dependencies and not claimed integrations**:

- **Bucket Robotics (YC S24)** — deployable manufacturing vision, edge hardware, fast iteration, synthetic/sample-data mindset.  
  https://www.ycombinator.com/companies/bucket-robotics

- **Allus AI (YC F25)** — manufacturing-focused vision models and rapid configuration for factory tasks.  
  https://www.ycombinator.com/companies/allus-ai

- **OctaPulse (YC W26)** — vision + edge computing + physical quality decisions/automation.  
  https://www.ycombinator.com/companies/octapulse

The project borrows the **engineering principles**: local inference, fast configuration, closed-loop decisions, and production observability. It does not copy branding, private APIs, or proprietary models.

## Quick start

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn garment_grader.api.main:app --reload
```

Run one simulated garment:

```bash
curl -X POST http://127.0.0.1:8000/v1/simulate \
  -H 'content-type: application/json' \
  -d '{"garment_id":"g-1001","customer":"retextil_demo"}'
```

Health endpoints:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/metrics
```

## Switch to Roboflow inference

Install the optional dependency:

```bash
pip install -e .[roboflow]
```

Run a self-hosted Roboflow Inference server, then set:

```bash
export GG_VISION_PROVIDER=roboflow
export GG_ROBOFLOW_API_URL=http://127.0.0.1:9001
export GG_ROBOFLOW_MODEL_ID=your-project/1
export ROBOFLOW_API_KEY=...
```

The adapter is in `src/garment_grader/integrations/roboflow_provider.py`.

## Projection idea

The project creates a **projection command**, not a permanent chemical mark.

A production line could use a calibrated DLP projector in a fixed downstream zone to show:

```text
GRADE B
ROUTE 04 →
STAIN ○
```

The code converts defect image coordinates into projector coordinates with a homography and calculates when a garment reaches the projection zone from encoder speed.

This is mainly useful for **human review, explainability and debugging**. A high-confidence fully automatic item can be routed directly without projection.

## What is real vs simulated

### Implemented

- FastAPI service
- Pydantic event contracts
- RGB frame-quality checks
- NIR quality checks
- provider interface
- deterministic local provider
- real Roboflow adapter
- grading + pricing + routing policy
- uncertainty / review gate
- projection geometry + timing
- local durable JSONL event queue
- optional Supabase sink
- Prometheus metrics
- Dockerfile
- tests

### Intentionally left for real hardware / model weights

- TensorRT engine implementation for the exact trained model
- industrial camera SDK integration
- real NIR spectrometer driver
- PLC fieldbus driver (OPC-UA / EtherCAT / vendor-specific)
- DLP hardware SDK
- production calibration data
- trained garment/defect model weights

Those parts require the real camera, sensor, conveyor, PLC and trained model. The interfaces are already present so they can be added without changing the business pipeline.

The repo also includes trainable **RGB multi-task ViT** and **NIR MLP** definitions plus an ONNX export script. They are architecture code only; no fake trained weights or accuracy claims are included.

## Repository map

```text
src/garment_grader/
  api/              HTTP API and health endpoints
  core/             pipeline, schemas, quality checks, policies
  integrations/     Roboflow + Modal examples
  models/           trainable RGB multi-task + NIR model definitions
  projector/        projection geometry and conveyor sync
  storage/          local JSONL + optional Supabase
  telemetry/        Prometheus metrics
configs/             customer grading/routing policies
contracts/           event schema
scripts/             demo / benchmark helpers
tests/               core unit tests
```

For the deeper technical reasoning, see `docs/ARCHITECTURE.md`.

## CI and latency checks

The repository includes GitHub Actions for compile + unit tests. A local benchmark script reports p50/p95/p99 for the pipeline plumbing:

```bash
PYTHONPATH=src python scripts/benchmark_local.py --n 200
```

Those mock numbers are **not** presented as TensorRT or Jetson performance. Real deployment acceptance must benchmark the exported model on the target device.
