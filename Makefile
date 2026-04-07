SUBDIRS := backend frontend
TARGETS := setup check test coverage

.PHONY: $(TARGETS) $(SUBDIRS)
$(TARGETS): $(SUBDIRS)
$(SUBDIRS):
	@$(MAKE) -C $@ $(MAKECMDGOALS)

NODE_MODULES := node_modules
TOUCH := node -e "import fs from 'fs'; const f=process.argv[1]; try{fs.utimesSync(f,new Date(),new Date())}catch{fs.closeSync(fs.openSync(f,'w'))}"

package-lock.json: package.json
	@echo "==> Updating lock file..."
	@npm install --package-lock-only

# Build node_modules with deps.
$(NODE_MODULES): package-lock.json
	@echo "==> Installing Node environment..."
	@npm install
	@$(TOUCH) $@

# Convenience target to build node_modules
.PHONY: setup
setup: $(NODE_MODULES)

.PHONY: check
check: $(NODE_MODULES)
	@echo "==> Linting docker compose files..."
	@npm run lint

COMPOSE_DEV = docker compose -f docker-compose.yml -f docker-compose.dev.yml

.PHONY: deploy
deploy:
	@echo "==> Pulling external images..."
	@docker compose pull --quiet
	@echo "==> Building images..."
	@docker compose build --pull
	@echo "==> Stopping and recreating services..."
	@docker compose up --detach --remove-orphans --wait
	@echo "==> Removing dangling images..."
	@docker image prune --force
	@echo "==> Deployment complete!"
	@docker compose ps

.PHONY: undeploy
undeploy:
	@echo "==> Stopping services..."
	@docker compose down
	@echo "==> Services stopped"

.PHONY: dev
dev:
	@$(COMPOSE_DEV) up -d
	@echo ""
	@echo "Application started! (development)"
	@echo "Frontend: http://localhost:5173"
	@echo "Backend: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

.PHONY: clean
clean:
	@echo "==> Cleaning ignored files..."
	@git clean -Xfd

.DEFAULT_GOAL := test
