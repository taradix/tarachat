NODE_MODULES := node_modules

NODE := npm exec -- node
TOUCH := $(NODE) -e 'fs.utimesSync(process.argv[1], new Date(), new Date())'

# Build node_modules with deps.
$(NODE_MODULES):
	@echo Installing Node environment
	@npm install
	@$(TOUCH) $@

# Convenience target to build node_modules
.PHONY: setup
setup: $(NODE_MODULES)

.PHONY: check
check: $(NODE_MODULES)
	@echo Linting code: Running eslint
	@npm run lint

.PHONY: test
test: $(NODE_MODULES)
	@echo Testing code: Running vitest
	@npm run test

.PHONY: coverage
coverage: $(NODE_MODULES)
	@echo Testing code: Running vitest --coverage
	@npm run coverage

.DEFAULT_GOAL := test
