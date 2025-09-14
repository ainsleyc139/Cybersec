# Cybersec Project Development

## Quick Start Commands

```bash
# Install all dependencies
npm run install:all

# Start both frontend and backend
npm run dev

# Build both for production
npm run build

# Run all tests
npm run test

# Lint all code
npm run lint

# Format all code
npm run format
```

## Development Workflow

### Starting Development
1. **Terminal 1 - Backend:**
   ```bash
   cd backend
   npm run dev
   ```

2. **Terminal 2 - Frontend:**
   ```bash
   cd frontend  
   npm run dev
   ```

3. **Or use the combined command:**
   ```bash
   npm run dev
   ```

### Making Changes
1. Make changes in respective directories
2. Both servers will hot-reload automatically
3. Test changes in browser at http://localhost:5173
4. Backend API available at http://localhost:3001

### Adding Features
1. **Frontend features:** Add components in `frontend/src/components/`
2. **Backend features:** Add routes in `backend/src/routes/`
3. **Shared types:** Define in respective `types/` directories

### Before Committing
```bash
# Run linting and tests
npm run lint
npm run test

# Format code
npm run format
```

## Project Architecture

```
cybersec/
├── 📱 frontend/           # React app
│   ├── src/
│   │   ├── components/    # Reusable components
│   │   ├── pages/        # Page components  
│   │   ├── services/     # API calls
│   │   └── types/        # TypeScript types
│   └── dist/             # Built files
├── 🔧 backend/            # Express API
│   ├── src/
│   │   ├── routes/       # API endpoints
│   │   ├── middleware/   # Custom middleware
│   │   ├── models/       # Data models
│   │   └── types/        # TypeScript types
│   └── dist/             # Compiled JS
└── 📚 docs/              # Documentation
```

## Key Features

### Security-First Design
- 🔐 JWT Authentication ready
- 🛡️ Security headers with Helmet
- 🚦 Rate limiting
- 🔍 Input validation
- 🔒 CORS configuration

### Developer Experience
- 🔥 Hot reload for both frontend and backend
- 📝 TypeScript for type safety
- 🎨 Tailwind CSS for styling
- 📏 ESLint and Prettier for code quality
- 🧪 Jest testing framework ready

### Production Ready
- 📦 Optimized builds
- 🏗️ CI/CD pipeline configured
- 📊 Health monitoring
- 📖 Comprehensive documentation

## Common Tasks

### Adding a New API Endpoint
1. Create route in `backend/src/routes/`
2. Add middleware if needed
3. Update types in `backend/src/types/`
4. Test with curl or Postman

### Adding a New Component
1. Create component in `frontend/src/components/`
2. Add types if needed
3. Export from index file
4. Use in pages or other components

### Database Integration (Future)
1. Choose database (PostgreSQL, MongoDB, etc.)
2. Add database client to backend dependencies
3. Create models in `backend/src/models/`
4. Add connection configuration

## Troubleshooting

### Port Already in Use
```bash
# Kill processes on ports 3001 and 5173
npx kill-port 3001 5173
```

### Dependencies Issues
```bash
# Clean install
npm run clean
npm run install:all
```

### Build Issues
```bash
# Clean builds
rm -rf frontend/dist backend/dist
npm run build
```

## Resources

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.com/docs/)
- [Express.js Guide](https://expressjs.com/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Cybersecurity Best Practices](./security.md)