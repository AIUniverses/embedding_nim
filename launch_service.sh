#!/bin/bash

# Enhanced Embedding Service Launcher
# Supports both simple and full deployment modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="simple"
ACTION="up"
BUILD=false
LOGS=false

# Help function
show_help() {
    echo -e "${BLUE}Enhanced Embedding Service Launcher${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE        Deployment mode: 'simple' or 'full' (default: simple)"
    echo "  -a, --action ACTION    Action: 'up', 'down', 'restart', 'logs', 'status' (default: up)"
    echo "  -b, --build           Force rebuild of containers"
    echo "  -l, --logs            Show logs after starting"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m simple -a up              # Start in simple mode"
    echo "  $0 -m full -a up -b             # Start in full mode with rebuild"
    echo "  $0 -m full -a logs              # Show logs for full mode"
    echo "  $0 -a down                      # Stop current deployment"
    echo ""
    echo "Modes:"
    echo "  simple: Only embedding service (lightweight)"
    echo "  full:   Embedding service + Redis + Monitoring + Vector DB"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -l|--logs)
            LOGS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "simple" && "$MODE" != "full" ]]; then
    echo -e "${RED}Error: Mode must be 'simple' or 'full'${NC}"
    exit 1
fi

# Set docker-compose file based on mode
if [[ "$MODE" == "simple" ]]; then
    COMPOSE_FILE="docker-compose.simple.yml"
else
    COMPOSE_FILE="docker-compose.yml"
fi

# Check if docker-compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo -e "${RED}Error: $COMPOSE_FILE not found${NC}"
    exit 1
fi

echo -e "${BLUE}Embedding Service - Mode: ${GREEN}$MODE${NC}"
echo -e "${BLUE}Action: ${GREEN}$ACTION${NC}"
echo -e "${BLUE}Compose file: ${GREEN}$COMPOSE_FILE${NC}"
echo ""

# Execute action
case $ACTION in
    up)
        echo -e "${GREEN}Starting embedding service...${NC}"
        
        if [[ "$BUILD" == true ]]; then
            echo -e "${YELLOW}Building containers...${NC}"
            docker-compose -f "$COMPOSE_FILE" build
        fi
        
        docker-compose -f "$COMPOSE_FILE" up -d
        
        echo -e "${GREEN}Service started successfully!${NC}"
        echo ""
        
        # Show service info
        echo -e "${BLUE}Service Information:${NC}"
        echo "- Embedding API: http://localhost:8009"
        echo "- Health Check: http://localhost:8009/v1/health/ready"
        echo "- API Documentation: http://localhost:8009/docs"
        
        if [[ "$MODE" == "full" ]]; then
            echo "- Redis: internal to the compose network only"
            echo "- Prometheus: http://localhost:9090"
            echo "- Grafana: http://localhost:3000 (admin / \$GRAFANA_ADMIN_PASSWORD)"
            echo "- Qdrant: http://localhost:6333"
        fi
        
        echo ""
        
        if [[ "$LOGS" == true ]]; then
            echo -e "${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
            docker-compose -f "$COMPOSE_FILE" logs -f
        fi
        ;;
        
    down)
        echo -e "${YELLOW}Stopping embedding service...${NC}"
        docker-compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}Service stopped successfully!${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}Restarting embedding service...${NC}"
        docker-compose -f "$COMPOSE_FILE" down
        
        if [[ "$BUILD" == true ]]; then
            echo -e "${YELLOW}Building containers...${NC}"
            docker-compose -f "$COMPOSE_FILE" build
        fi
        
        docker-compose -f "$COMPOSE_FILE" up -d
        echo -e "${GREEN}Service restarted successfully!${NC}"
        ;;
        
    logs)
        echo -e "${BLUE}Showing logs for $MODE mode...${NC}"
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
        
    status)
        echo -e "${BLUE}Service Status:${NC}"
        docker-compose -f "$COMPOSE_FILE" ps
        echo ""
        
        # Check service health
        echo -e "${BLUE}Health Checks:${NC}"
        
        # Check embedding service
        if curl -s http://localhost:8009/v1/health/ready > /dev/null 2>&1; then
            echo -e "- Embedding Service: ${GREEN}✓ Healthy${NC}"
        else
            echo -e "- Embedding Service: ${RED}✗ Unhealthy${NC}"
        fi
        
        if [[ "$MODE" == "full" ]]; then
            # Check Redis
            if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; then
                echo -e "- Redis: ${GREEN}✓ Healthy${NC}"
            else
                echo -e "- Redis: ${RED}✗ Unhealthy${NC}"
            fi
            
            # Check Prometheus
            if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
                echo -e "- Prometheus: ${GREEN}✓ Healthy${NC}"
            else
                echo -e "- Prometheus: ${RED}✗ Unhealthy${NC}"
            fi
            
            # Check Grafana
            if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
                echo -e "- Grafana: ${GREEN}✓ Healthy${NC}"
            else
                echo -e "- Grafana: ${RED}✗ Unhealthy${NC}"
            fi
        fi
        ;;
        
    *)
        echo -e "${RED}Error: Unknown action '$ACTION'${NC}"
        echo "Valid actions: up, down, restart, logs, status"
        exit 1
        ;;
esac
