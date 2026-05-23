# Production build for the GPR frontend.
# Multi-stage: build with Node, serve with nginx.

# ── Stage 1: build ───────────────────────────────────────────────
FROM node:20-alpine AS build
WORKDIR /app

# Install deps separately from source for cache efficiency
COPY package*.json ./
RUN npm ci

COPY . .
# VITE_API_URL is baked into the bundle at build time
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# ── Stage 2: serve ───────────────────────────────────────────────
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
