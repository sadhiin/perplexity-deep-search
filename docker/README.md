# Docker Setup for Perplexity Deep Search

This directory contains Docker configuration files to containerize both the backend and frontend of the Perplexity Deep Search application.

## Architecture

- **Backend**: Python/FastAPI application using UV package manager
- **Frontend**: Static HTML/CSS/JS served by Nginx
- **Networking**: Services communicate through a dedicated Docker network

## Files

- `Dockerfile.backend` - Backend container configuration
- `Dockerfile.frontend` - Frontend container configuration
- `nginx.conf` - Nginx configuration for frontend
- `docker-compose.yml` - Multi-service orchestration

## Quick Start

1. **Build and run all services:**

   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Frontend: `http://localhost`
   - Backend API: `http://localhost:8000`

3. **Stop services:**

   ```bash
   docker-compose down
   ```

## Development

### Backend Development

The backend container mounts the source code as a volume, so changes are reflected immediately:

```bash
# Run only backend for development
docker-compose up backend

# View logs
docker-compose logs -f backend
```

### Frontend Development

The frontend also uses volume mounting for hot reloading:

```bash
# Run only frontend for development
docker-compose up frontend

# View logs
docker-compose logs -f frontend
```

## Environment Variables

### Backend

- `PYTHONUNBUFFERED=1` - Disable Python output buffering
- `PYTHONDONTWRITEBYTECODE=1` - Prevent .pyc file generation

### Frontend

- `API_BASE_URL=http://backend:8000` - Backend API endpoint (internal Docker network)

## Networking

Services communicate through the `perplexity-network` bridge network:

- Backend is accessible as `backend:8000` from frontend
- External access through port mappings (80 for frontend, 8000 for backend)

## Health Checks

Both services include health check endpoints:

- Backend: `GET /health`
- Frontend: `GET /health`

## Volumes

- `./:/app` - Mounts source code for backend development
- `./frontend:/usr/share/nginx/html` - Mounts frontend files
- `./docker/nginx.conf:/etc/nginx/nginx.conf` - Mounts nginx configuration

## Production Deployment

For production deployment:

1. **Build optimized images:**

   ```bash
   docker-compose -f docker-compose.yml up --build -d
   ```

2. **Use environment-specific configurations:**

   ```bash
   # Create production override file
   cp docker-compose.yml docker-compose.prod.yml
   # Edit docker-compose.prod.yml with production settings
   ```

3. **Set production environment variables:**

   ```bash
   export API_BASE_URL=https://api.yourdomain.com
   docker-compose -f docker-compose.prod.yml up -d
   ```

## Troubleshooting

### Common Issues

1. **Port conflicts:**

   ```bash
   # Check what's using ports
   netstat -tulpn | grep :80
   netstat -tulpn | grep :8000

   # Change ports in docker-compose.yml if needed
   ```

2. **Permission issues:**

   ```bash
   # Ensure proper file permissions
   sudo chown -R $USER:$USER .
   ```

3. **Build failures:**

   ```bash
   # Clear Docker cache
   docker system prune -a

   # Rebuild without cache
   docker-compose build --no-cache
   ```

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs backend
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f

# View container status
docker-compose ps
```

## Security Considerations

- Nginx configured with security headers
- Content Security Policy enabled
- X-Frame-Options and X-Content-Type-Options set
- Minimal base images used (alpine, slim)
- Non-root user execution where possible

## Performance

- UV package manager for fast Python dependency installation
- Nginx with gzip compression enabled
- Static asset caching headers
- Health checks for service monitoring
