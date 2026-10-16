import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // In dev, talk to the backend started with `uvicorn app.main:app --reload`.
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
