# Backend Documentation

## Overview
The backend is built with Node.js, Express, and TypeScript, following security best practices for cybersecurity applications.

## Technology Stack
- **Node.js** - JavaScript runtime environment
- **Express** - Web application framework
- **TypeScript** - Type safety and better development experience
- **JWT** - JSON Web Tokens for authentication
- **bcrypt** - Password hashing
- **Helmet** - Security headers
- **CORS** - Cross-origin resource sharing
- **Rate Limiting** - Protection against abuse

## Project Structure
```
backend/
├── src/
│   ├── routes/        # API route handlers
│   ├── middleware/    # Custom middleware functions
│   ├── models/        # Data models
│   ├── services/      # Business logic services
│   ├── utils/         # Utility functions
│   ├── types/         # TypeScript type definitions
│   └── index.ts       # Application entry point
├── dist/             # Compiled JavaScript (generated)
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

2. Copy environment variables:
   ```bash
   cp .env.example .env
   ```

3. Update `.env` with your configuration

4. Start development server:
   ```bash
   npm run dev
   ```

5. Server runs on `http://localhost:3001`

### Available Scripts
- `npm run dev` - Start development server with watch mode
- `npm run build` - Compile TypeScript to JavaScript
- `npm start` - Start production server
- `npm run test` - Run tests
- `npm run lint` - Run ESLint

## API Endpoints

### Health Check
- `GET /health` - Server health status

### API Info
- `GET /api` - API information and available endpoints

### Authentication (Future)
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/profile` - Get user profile

## Security Features

### Implemented
- **Helmet** - Security headers (XSS, CSRF, etc.)
- **CORS** - Configured for frontend domain
- **Rate Limiting** - 100 requests per 15 minutes per IP
- **Input Validation** - Request validation middleware
- **Error Handling** - Secure error messages

### Planned
- JWT authentication
- Password hashing with bcrypt
- Input sanitization
- SQL injection prevention
- Session management
- Audit logging

## Environment Variables
Required environment variables in `.env`:
```
NODE_ENV=development
PORT=3001
FRONTEND_URL=http://localhost:5173
JWT_SECRET=your-super-secret-jwt-key
JWT_EXPIRES_IN=24h
BCRYPT_ROUNDS=12
```

## Error Handling
- Development: Detailed error messages
- Production: Generic error messages
- All errors are logged
- HTTP status codes follow REST standards

## Database Integration (Future)
The backend is prepared for database integration:
- Support for PostgreSQL, MySQL, or MongoDB
- Data models defined in `/src/models`
- Connection pooling for performance
- Migration scripts for schema management

## Deployment
1. Build the application:
   ```bash
   npm run build
   ```

2. Start production server:
   ```bash
   npm start
   ```

3. Use process manager like PM2 for production:
   ```bash
   pm2 start dist/index.js --name cybersec-backend
   ```

## Testing
- Unit tests with Jest
- Integration tests for API endpoints
- Security tests for vulnerabilities
- Load testing for performance

## Monitoring
- Health check endpoint for uptime monitoring
- Request logging middleware
- Error tracking and alerting
- Performance metrics collection