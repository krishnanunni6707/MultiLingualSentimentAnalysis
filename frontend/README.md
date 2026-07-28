# Frontend — Multilingual Sentiment Analysis UI

A Next.js 16 web interface for the multilingual sentiment analysis backend. Built with TailwindCSS and TypeScript.

---

## Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router, Turbopack) |
| Language | TypeScript |
| Styling | TailwindCSS v4 |
| Icons | lucide-react |
| Runtime | React 19 |

---

## Quick Start

```bash
npm install
npm run dev -- --port 3001
# → http://localhost:3001
```

Backend must be running at `http://localhost:8000` (see `backend/README.md`).

---

## Features

- **Auto language detection** — input Hindi or English, the backend identifies it automatically
- **Expandable input bubble** — shows submitted text preview; click to expand full text
- **Animated 3-step progress** — "Extracting & Segmenting" → "Running Dual Model Inference" → "Generating Summary"
- **Results panel** — dominant emotion badge, polarity score ring, full emotion breakdown from `overall_sentiment.emotions`
- **Cross-check display** — shows `agreement_rate`, `primary_source`, and translated text when available
- **Summary section** — displays Llama-generated summary if `summarizer_status: "active"`
- **Text-to-Speech** — calls `/api/v1/speak` (Edge-TTS) for audio playback

---

## Project Structure

```
frontend/
├── app/
│   ├── page.tsx          ← Main page: input, progress, results
│   ├── layout.tsx        ← Root layout + metadata
│   └── globals.css       ← Global styles
├── public/               ← Static assets
├── next.config.ts        ← API proxy config (proxies /api → localhost:8000)
├── package.json
└── tsconfig.json
```

---

## Environment

The Next.js config proxies `/api/*` requests to the backend:

```typescript
// next.config.ts
rewrites: async () => [
  { source: '/api/:path*', destination: 'http://localhost:8000/api/:path*' }
]
```

No `.env` needed for local development.

---

## Scripts

```bash
npm run dev      # Start development server (Turbopack)
npm run build    # Production build
npm run start    # Start production server
npm run lint     # ESLint
```
