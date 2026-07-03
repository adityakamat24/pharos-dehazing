# Dehazing & Desmoking — Real-Time, Unified

Research project: a novel, real-time image/video dehazing **and** smoke-removal method.

Target use-cases:
- **Seeing through smoke** — firefighting, search & rescue (the life-saving case)
- **Atmospheric haze** — photography, driving, public webcams
- **Satellite / remote-sensing haze** — thin cloud & haze over imagery

Status: research phase — method design in progress. This README will be replaced by the full
method description once the design is finalized.

## Environment

- Local: Windows, RTX 3060 6GB (prototyping, sanity training, real-time inference)
- Cloud: vast.ai GPU pod (full training)
- Python 3.14, PyTorch (CUDA) in `.venv/`
