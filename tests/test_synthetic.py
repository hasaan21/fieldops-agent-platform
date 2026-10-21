from capstone.data.synthetic.generator import Generator


def small_world():
    return Generator(seed=1, customers_per_tenant=6, jobs_per_tenant=30,
                     transcripts_per_tenant=40).build()


def test_generation_is_deterministic():
    a, b = small_world(), small_world()
    assert a.transcripts == b.transcripts
    assert a.jobs == b.jobs


def test_every_row_is_tenant_scoped():
    w = small_world()
    tenant_ids = {t["tenant_id"] for t in w.tenants}
    for name, rows in w.tables().items():
        if name == "tenants":
            continue
        for row in rows:
            assert row["tenant_id"] in tenant_ids, name


def test_labels_are_consistent():
    w = small_world()
    for tx in w.transcripts:
        lab = tx["labels"]
        assert not (lab["should_book"] and lab["should_escalate"])
        assert not (lab["should_book"] and lab["needs_clarification"])
        if lab["failure_mode"] == "contains_pii":
            assert lab["must_not_contain"]
        if lab["should_book"]:
            assert "create_job" in lab["expected_tools"]


def test_tenant_facts_differ():
    w = small_world()
    fees = {t["callout_fee_gbp"] for t in w.tenants}
    assert len(fees) == len(w.tenants)
