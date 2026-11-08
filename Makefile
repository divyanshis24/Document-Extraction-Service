# Every check in one place. `make check` runs them all, in the order a release would.
#
#   make test       unit tests, no database, no network, a few seconds
#   make test-db    the same plus the Postgres-backed queue, worker and API tests (throwaway container)
#   make eval       accuracy on the sample set, cached and spend-capped (needs LLM_API_KEY in .env)
#   make up         build and start the service, wait for /health
#   make samples    the sample set through the running service -> outputs/, then the validator
#   make load       300 synthetic jobs through the service with the stub extractor, one worker killed mid-run
#   make check      test-db + up + samples + load
#   make down       stop the service

PY ?= .venv/bin/python
TEST_PG_NAME := extraction-test-pg
TEST_PG_PORT ?= 5433
export TEST_DATABASE_URL ?= postgresql://postgres:pg@localhost:$(TEST_PG_PORT)/postgres

.PHONY: test test-db eval up down samples load check wait-health

test:
	$(PY) -m pytest -q

test-db:
	@docker rm -f $(TEST_PG_NAME) >/dev/null 2>&1 || true
	docker run -d --rm --name $(TEST_PG_NAME) -e POSTGRES_PASSWORD=pg -p $(TEST_PG_PORT):5432 postgres:16-alpine >/dev/null
	@until docker exec $(TEST_PG_NAME) pg_isready -h 127.0.0.1 -U postgres >/dev/null 2>&1; do sleep 0.5; done
	$(PY) -m pytest -q; status=$$?; docker rm -f $(TEST_PG_NAME) >/dev/null; exit $$status

eval:
	$(PY) scripts/eval.py --details

up:
	docker compose up --build -d
	$(MAKE) wait-health

wait-health:
	@for i in $$(seq 1 60); do \
	  if curl -fs http://localhost:8080/health >/dev/null 2>&1; then echo "healthy after $$i s"; exit 0; fi; sleep 1; \
	done; echo "service did not become healthy in 60s"; docker compose logs --tail=50 api; exit 1

down:
	docker compose down

samples: wait-health
	$(PY) scripts/load.py samples

# Workers restart with the stub extractor (no model calls, 5s per job), then go back to the real one.
load:
	EXTRACTOR=stub STUB_EXTRACTOR_DELAY_SECONDS=5 docker compose up -d --no-deps --force-recreate --scale worker=2 worker
	@sleep 4
	$(PY) scripts/load.py synthetic -n 200 --kill-worker 20; status=$$?; \
	docker compose up -d --no-deps --force-recreate --scale worker=2 worker; exit $$status

check: test-db up samples load
	@echo "all checks passed"
