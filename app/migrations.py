"""Schema migrations: an ordered, append-only list applied in one transaction."""
from __future__ import annotations

_ADVISORY_LOCK_ID = 7203114

MIGRATIONS: list[str] = [
    """
    CREATE TABLE documents (
        sha256      text PRIMARY KEY,
        content     bytea NOT NULL,
        size_bytes  bigint NOT NULL,
        created_at  timestamptz NOT NULL DEFAULT now()
    );
    -- PDFs and Office files are already compressed; don't spend CPU recompressing.
    ALTER TABLE documents ALTER COLUMN content SET STORAGE EXTERNAL;

    CREATE TABLE jobs (
        id                 uuid PRIMARY KEY,
        org_id             text NOT NULL,
        status             text NOT NULL DEFAULT 'PENDING'
                           CHECK (status IN ('PENDING','RUNNING','SUCCESS','PARTIAL','FAILED')),
        stage              text,
        document_sha256    text NOT NULL REFERENCES documents (sha256),
        file_name          text NOT NULL,
        file_type          text NOT NULL CHECK (file_type IN ('pdf','docx','xlsx','csv')),
        field_list         jsonb NOT NULL,
        field_list_sha256  text NOT NULL,
        result             jsonb,
        error              jsonb,
        attempts           integer NOT NULL DEFAULT 0,
        lease_token        uuid,
        lease_expires_at   timestamptz,
        created_at         timestamptz NOT NULL DEFAULT now(),
        updated_at         timestamptz NOT NULL DEFAULT now(),
        finished_at        timestamptz,
        CONSTRAINT result_iff_success CHECK ((status IN ('SUCCESS','PARTIAL')) = (result IS NOT NULL)),
        CONSTRAINT error_iff_failed   CHECK ((status = 'FAILED') = (error IS NOT NULL)),
        CONSTRAINT lease_iff_running  CHECK ((status = 'RUNNING') = (lease_token IS NOT NULL))
    );

    -- The database, not application code, guarantees one live job per document and
    -- field list within a tenant, even under simultaneous submissions. FAILED jobs
    -- are excluded so a failed document can be resubmitted.
    CREATE UNIQUE INDEX jobs_dedupe ON jobs (org_id, document_sha256, field_list_sha256)
        WHERE status <> 'FAILED';
    CREATE INDEX jobs_pending ON jobs (created_at) WHERE status = 'PENDING';
    CREATE INDEX jobs_running ON jobs (lease_expires_at) WHERE status = 'RUNNING';

    CREATE TABLE job_webhooks (
        job_id      uuid NOT NULL REFERENCES jobs (id),
        url         text NOT NULL,
        created_at  timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (job_id, url)
    );

    -- Last line of defence for "terminal statuses never change", whatever a bug
    -- or a stale worker tries to do.
    CREATE FUNCTION jobs_terminal_is_immutable() RETURNS trigger AS $$
    BEGIN
        IF OLD.status IN ('SUCCESS','PARTIAL','FAILED') THEN
            RAISE EXCEPTION 'job % is terminal (%) and cannot change', OLD.id, OLD.status
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER jobs_terminal_guard BEFORE UPDATE ON jobs
        FOR EACH ROW EXECUTE FUNCTION jobs_terminal_is_immutable();
    """,
    """
    -- Delivery state lives here, not on jobs, so it can change after a job is terminal.
    ALTER TABLE job_webhooks
        ADD COLUMN attempts        integer NOT NULL DEFAULT 0,
        ADD COLUMN next_attempt_at timestamptz NOT NULL DEFAULT now(),
        ADD COLUMN delivered_at    timestamptz,
        ADD COLUMN gave_up_at      timestamptz,
        ADD COLUMN last_error      text;
    CREATE INDEX job_webhooks_due ON job_webhooks (next_attempt_at)
        WHERE delivered_at IS NULL AND gave_up_at IS NULL;

    -- When one worker is rate limited, every worker holds back on that model.
    CREATE TABLE llm_cooldowns (
        model  text PRIMARY KEY,
        until  timestamptz NOT NULL
    );
    """,
]


async def apply_migrations(conn) -> None:
    """Safe to call from every API and worker process at the same time."""
    async with conn.transaction():
        # Serialises concurrent startups and is released when the transaction ends.
        await conn.execute("SELECT pg_advisory_xact_lock(%s)", (_ADVISORY_LOCK_ID,))
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cur = await conn.execute("SELECT coalesce(max(version), 0) AS v FROM schema_migrations")
        current = (await cur.fetchone())["v"]
        for version, sql in enumerate(MIGRATIONS, start=1):
            if version > current:
                await conn.execute(sql)
                await conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
