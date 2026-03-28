# Tempest Embedding

Temporal link prediction on dynamic graphs, powered by [Tempest](https://github.com/ashfaq1701/temporal-random-walk) for walk generation.

## What this does

Tempest Embedding learns node representations from temporal graphs -- graphs where edges carry timestamps and arrive over time. Given a history of interactions, the model predicts future links between nodes.

Each node's embedding is built from multiple temporal random walks rooted at that node. Walks are encoded using a GRU with continuous-time ODE evolution, capturing both the sequence of neighbors visited and the timing of interactions. Walk embeddings are aggregated into a single node representation, and the model is trained with a contrastive objective that scores true edges higher than random negatives.

## How it works

The system processes temporal graphs in chronological order. Training edges are streamed into the walk engine in time-duration windows, so the model sees the graph grow incrementally. At each step, walks are generated from the current graph state, and the model trains on the edges from that window. This makes the training regime naturally temporal -- earlier windows see less of the graph, and later windows see more.

Walk generation is handled by Tempest, a GPU-native temporal random walk engine. This replaces the tree-expansion approach used in prior work (NeurTWs), which was CPU-bound and dominated training time.

## Supported datasets

CollegeMsg, Enron, TaobaoSmall, MOOC, Wikipedia, Reddit.

## Quick start

```bash
# From the scripts/ directory
bash collegemsg_transductive.sh
```
