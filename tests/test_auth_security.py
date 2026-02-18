import os
import stat
import json
import pytest
from unittest.mock import patch
from tensorcircuit.cloud.apis import set_token, saved_token

def test_token_file_permissions(tmp_path):
    # Mock os.path.expanduser to return tmp_path
    # We patch where it is used in tensorcircuit.cloud.apis
    with patch("tensorcircuit.cloud.apis.os.path.expanduser", return_value=str(tmp_path)):
        authpath = tmp_path / ".tc.auth.json"

        # Ensure clean state for saved_token global variable if necessary
        # set_token with clear=True clears the global saved_token
        set_token(clear=True)

        # Scenario 1: File creation (new file)
        # Set a dummy token
        set_token(token="dummy_token_1", provider="tencent", cached=True)

        # Verify file exists
        assert authpath.exists()

        # Verify permissions (only on POSIX systems where these bits are meaningful)
        if os.name == "posix":
            st = os.stat(authpath)
            # Check that group and others have no permissions (should be 0)
            assert (st.st_mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
                                  stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)) == 0

        # Scenario 2: File update (existing file with insecure permissions)
        if os.name == "posix":
            # Manually set insecure permissions to simulate an existing insecure file
            os.chmod(authpath, 0o666)
            st_before = os.stat(authpath)
            # Verify it is indeed insecure (readable by others)
            assert (st_before.st_mode & stat.S_IROTH)

        # Update token (add another token or update existing)
        # This triggers the write logic again
        set_token(token="dummy_token_2", provider="local", cached=True)

        # Verify permissions again (should be fixed to 0600)
        if os.name == "posix":
            st_after = os.stat(authpath)
            assert (st_after.st_mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
                                       stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)) == 0

        # Verify content updated and readable
        with open(authpath, "r") as f:
            data = json.load(f)
            assert data is not None
            # Check if both tokens are present (implementation detail: set_token updates the dict)
            # Keys are typically "provider::device" or "provider::"
            # Since we added "tencent" and "local", we expect keys for them.
            # But the exact key format depends on set_token logic.
            # We are mainly testing permissions here.

if __name__ == "__main__":
    # Manually running with pytest main if executed as script
    import sys
    sys.exit(pytest.main([__file__]))
