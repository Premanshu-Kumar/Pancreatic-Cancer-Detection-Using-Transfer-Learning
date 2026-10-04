# PancreaAI Frontend

Production-oriented Next.js frontend for the Pancreatic Cancer Detection Using Transfer Learning project.

## Run

```bash
npm install
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_URL` to the FastAPI server, normally `http://localhost:8000`.

## Real API integration

The UI uses the existing backend contracts:

- `GET /api/models`
- `GET /api/health`
- `POST /api/predict`
- `POST /api/predict/batch` (ready in the API client for future multi-slice UI)
- `GET /api/report/download`

No prediction values are mocked. The 3D anatomy is an explicitly labelled visualization and is not used for diagnosis.

## Integration

Place this directory at `frontend/` in the repository. Run it separately from the FastAPI service during development. For production, either deploy the Next.js frontend separately and set `NEXT_PUBLIC_API_URL` to the API origin, or export/serve it behind the same origin according to the deployment architecture.
