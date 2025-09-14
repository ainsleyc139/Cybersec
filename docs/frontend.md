# Frontend Documentation

## Overview
The frontend is built with React 18, TypeScript, and Vite for fast development and building.

## Technology Stack
- **React 18** - Modern React with concurrent features
- **TypeScript** - Type safety and better development experience
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **Axios** - HTTP client for API requests

## Project Structure
```
frontend/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/         # Page components
│   ├── hooks/         # Custom React hooks
│   ├── services/      # API service functions
│   ├── types/         # TypeScript type definitions
│   ├── utils/         # Utility functions
│   └── styles/        # Global styles and themes
├── public/            # Static assets
└── package.json
```

## Development

### Prerequisites
- Node.js 18+
- npm or yarn

### Setup
1. Install dependencies:
   ```bash
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Open browser to `http://localhost:5173`

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run test` - Run tests

## Security Considerations
- All user inputs are sanitized
- API requests include proper authentication headers
- Sensitive data is never stored in localStorage
- CSP headers are configured for XSS protection
- HTTPS is enforced in production

## API Integration
The frontend communicates with the backend through REST APIs:
- Base URL: `http://localhost:3001` (development)
- Authentication: JWT tokens
- Error handling: Centralized error boundary

## Deployment
1. Build the application:
   ```bash
   npm run build
   ```

2. The `dist/` folder contains the production-ready files

3. Deploy to your preferred hosting service (Netlify, Vercel, etc.)

## Environment Variables
Create a `.env` file in the frontend directory:
```
VITE_API_BASE_URL=http://localhost:3001
VITE_APP_TITLE=Cybersec Project
```