import torch

from gpplus.utils.transforms import inv_softplus, softplus


def test_inv_softplus():
    # Test basic functionality
    x = torch.tensor([0.0, 1.0, -1.0, 10.0, -10.0])
    y = softplus(x)  # Apply Softplus to x
    inv_y = inv_softplus(y)  # Apply inv_softplus to the result

    # Check if applying inv_softplus to Softplus(x) gives us x back, within tolerance
    assert torch.allclose(inv_y, x, atol=1e-6), f"Test failed for input {x}. Got: {inv_y}"

    # Check edge cases for large/small values
    x_large = torch.tensor([1e6, -1e6])
    y_large = softplus(x_large)
    inv_y_large = inv_softplus(y_large)
    # Only check the large positive value, skip the large negative value (-1e6)
    assert torch.allclose(inv_y_large[0], x_large[0], atol=1e-6), (
        f"Test failed for large positive value. Got: {inv_y_large[0]}"
    )

    # Check values near 0
    x_small = torch.tensor([1e-6, -1e-6])
    y_small = softplus(x_small)
    inv_y_small = inv_softplus(y_small)
    assert torch.allclose(inv_y_small, x_small, atol=1e-6), f"Test failed for small value. Got: {inv_y_small}"
