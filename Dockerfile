# syntax=docker/dockerfile:1.6
#
# Low-latency build: fd-passing Rust LB + epoll API. Target = linux/amd64 + AVX2.
# One image carries all binaries; compose selects `lb` or `api` per service.

#######################################################################
# Stage 1 — build api, lb, builder with AVX2 codegen.
#######################################################################
FROM --platform=linux/amd64 rust:1-bookworm AS build
ENV RUSTFLAGS="-C target-cpu=haswell"
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --bins

#######################################################################
# Stage 2 — build the IVF index (int16, 2048 centroids).
#######################################################################
FROM --platform=linux/amd64 build AS index
ARG CENTROIDS=2048
COPY resources/references.json.gz /resources/references.json.gz
RUN mkdir -p /output \
 && CENTROIDS=${CENTROIDS} /app/target/release/builder /resources/references.json.gz /output/index.bin \
 && ls -la /output

#######################################################################
# Stage 3 — minimal runtime with both binaries + the index.
#######################################################################
FROM --platform=linux/amd64 debian:bookworm-slim AS api
LABEL org.opencontainers.image.source="https://github.com/bmtec/rinha-backend-2026" \
      org.opencontainers.image.licenses="MIT"
COPY --from=build /app/target/release/api /usr/local/bin/api
COPY --from=build /app/target/release/lb  /usr/local/bin/lb
COPY --from=index /output/index.bin /data/index.bin
RUN mkdir -p /sockets
ENV INDEX_PATH=/data/index.bin NPROBE=10
# No ENTRYPOINT — docker-compose selects `api` or `lb` per service.
