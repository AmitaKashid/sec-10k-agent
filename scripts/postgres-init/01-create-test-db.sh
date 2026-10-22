#!/bin/bash
# Auto-create the integration-test database on first Postgres start.
#
# Mounted into /docker-entrypoint-initdb.d/ by docker-compose.yml.
# Only runs ONCE per volume — on first init. Existing volumes (e.g. an
# already-running `sec_agent_postgres` container) are unaffected; for
# those, the integration test conftest's `_ensure_test_db_exists()`
# helper creates `sec_agent_test` on the first pytest run.
#
# This script exists so a fresh `docker compose up -d postgres` on a
# clean machine yields a working test database with zero manual steps.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE sec_agent_test'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'sec_agent_test')
    \gexec
EOSQL
