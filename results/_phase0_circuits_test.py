def test_parameterized_output_changes_with_theta():
    import jax, jax.numpy as jnp
    from results._phase0_circuits import verify_dynamic

    r = verify_dynamic(jnp.array([0.7] * 64), jnp.array([0.9] * 64), n=8, depth=2)
    assert r["output_changes"] is True
    assert r["hlo_has_runtime_param"] is True


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__, "-v"]))
