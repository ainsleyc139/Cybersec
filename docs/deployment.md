# Deployment Guide

## Overview
This guide covers deployment strategies for both frontend and backend components of the Cybersec project.

## Deployment Architecture

### Option 1: Separate Deployment
- **Frontend**: Deploy to Netlify, Vercel, or GitHub Pages
- **Backend**: Deploy to Heroku, Railway, or DigitalOcean

### Option 2: Unified Deployment
- **Full Stack**: Deploy both to a VPS or cloud provider
- **Docker**: Containerized deployment with Docker Compose

## Frontend Deployment

### Netlify (Recommended for Static Sites)
1. Build the frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. Deploy the `dist/` folder to Netlify
3. Configure environment variables in Netlify dashboard
4. Set up custom domain if needed

### Vercel
1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Deploy from frontend directory:
   ```bash
   cd frontend
   vercel --prod
   ```

### GitHub Pages
1. Add deployment script to package.json:
   ```json
   {
     "scripts": {
       "deploy": "gh-pages -d dist"
     }
   }
   ```

2. Build and deploy:
   ```bash
   npm run build
   npm run deploy
   ```

## Backend Deployment

### Heroku
1. Create Heroku app:
   ```bash
   heroku create cybersec-backend
   ```

2. Set environment variables:
   ```bash
   heroku config:set NODE_ENV=production
   heroku config:set JWT_SECRET=your-production-secret
   ```

3. Deploy:
   ```bash
   git push heroku main
   ```

### Railway
1. Connect GitHub repository to Railway
2. Configure environment variables
3. Railway automatically deploys on git push

### DigitalOcean App Platform
1. Create new app from GitHub repository
2. Configure build and run commands:
   - Build: `cd backend && npm install && npm run build`
   - Run: `cd backend && npm start`
3. Set environment variables in dashboard

## Docker Deployment

### Docker Compose (Recommended)
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_BASE_URL=http://localhost:3001
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "3001:3001"
    environment:
      - NODE_ENV=production
      - PORT=3001
      - JWT_SECRET=${JWT_SECRET}
    env_file:
      - ./backend/.env
```

Deploy with:
```bash
docker-compose up -d
```

### Individual Dockerfiles

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

**Backend Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3001
CMD ["npm", "start"]
```

## Environment Configuration

### Production Environment Variables

**Frontend (.env.production):**
```
VITE_API_BASE_URL=https://your-backend-domain.com
VITE_APP_TITLE=Cybersec Project
```

**Backend (.env.production):**
```
NODE_ENV=production
PORT=3001
FRONTEND_URL=https://your-frontend-domain.com
JWT_SECRET=your-super-secure-production-secret
JWT_EXPIRES_IN=24h
BCRYPT_ROUNDS=12
```

## SSL/TLS Configuration

### Let's Encrypt (Free SSL)
```bash
# Install certbot
sudo apt install certbot

# Get certificate
sudo certbot certonly --standalone -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 2 * * * /usr/bin/certbot renew --quiet
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Backend API
    location /api {
        proxy_pass http://localhost:3001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring and Maintenance

### Health Checks
- Frontend: Monitor build status and deployment
- Backend: Use `/health` endpoint for uptime monitoring

### Logging
- Set up centralized logging (e.g., Loggly, Papertrail)
- Monitor error rates and performance metrics

### Backups
- Database backups (if applicable)
- Environment configuration backups
- Code repository backups

## Security Considerations

### Production Checklist
- [ ] Change all default passwords and secrets
- [ ] Enable HTTPS everywhere
- [ ] Configure proper CORS origins
- [ ] Set up rate limiting
- [ ] Enable security headers
- [ ] Regular security updates
- [ ] Monitor for vulnerabilities
- [ ] Set up intrusion detection

### Environment Security
- [ ] Use environment variables for secrets
- [ ] Restrict database access
- [ ] Configure firewall rules
- [ ] Regular security audits
- [ ] Access logging and monitoring