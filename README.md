# Cybersec School Project

A comprehensive cybersecurity project with frontend and backend components designed for educational purposes.

## Project Structure

```
cybersec/
├── frontend/          # React TypeScript frontend
├── backend/           # Node.js Express TypeScript backend
├── docs/             # Project documentation
├── .github/          # GitHub workflows and templates
└── README.md         # This file
```

## Quick Start

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cybersec
   ```

2. **Install dependencies**
   ```bash
   # Install root dependencies
   npm install
   
   # Install frontend dependencies
   cd frontend && npm install && cd ..
   
   # Install backend dependencies
   cd backend && npm install && cd ..
   ```

3. **Start development servers**
   ```bash
   # Start both frontend and backend concurrently
   npm run dev
   
   # Or start individually:
   npm run dev:frontend   # Frontend only
   npm run dev:backend    # Backend only
   ```

## Available Scripts

- `npm run dev` - Start both frontend and backend in development mode
- `npm run dev:frontend` - Start frontend development server
- `npm run dev:backend` - Start backend development server
- `npm run build` - Build both frontend and backend for production
- `npm run test` - Run tests for both frontend and backend
- `npm run lint` - Lint all code
- `npm run format` - Format all code with Prettier

## Technology Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **React Router** for navigation
- **Axios** for API communication

### Backend
- **Node.js** with Express
- **TypeScript** for type safety
- **JWT** for authentication
- **bcrypt** for password hashing
- **Helmet** for security headers
- **CORS** for cross-origin requests

### Development Tools
- **ESLint** for code linting
- **Prettier** for code formatting
- **Husky** for git hooks
- **Jest** for testing

## Security Features

This project includes several cybersecurity best practices:

- Input validation and sanitization
- Secure password hashing
- JWT-based authentication
- Security headers with Helmet
- CORS configuration
- Rate limiting
- SQL injection prevention
- XSS protection

## Project Workflow

### Development Branches
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Individual feature branches
- `frontend/*` - Frontend-specific features
- `backend/*` - Backend-specific features

### Merging Strategy
1. Develop features in separate branches
2. Frontend and backend can be developed independently
3. Merge to `develop` branch for integration testing
4. Merge to `main` for production deployment

## Documentation

- [Frontend Documentation](./docs/frontend.md)
- [Backend Documentation](./docs/backend.md)
- [API Documentation](./docs/api.md)
- [Security Guidelines](./docs/security.md)
- [Deployment Guide](./docs/deployment.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is for educational purposes as part of a cybersecurity school project.