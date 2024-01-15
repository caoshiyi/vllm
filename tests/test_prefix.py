from vllm.prefix import PrefixPool

import pytest

@pytest.fixture(scope='module')
def no_max_capacity_prefix_pool() -> PrefixPool:
    return PrefixPool(block_size=32)
                      

def test_prefix_pool_no_max_capacity(no_max_capacity_prefix_pool: PrefixPool):
    # Test that the same prefix produces the same object even after
    # multiple repetitions of the object, meaning that no new prefix object
    # is created, but the one already stored is returned
    prefix_1 = no_max_capacity_prefix_pool.add_or_get_prefix([1, 2, 3])
    prefix_2 = no_max_capacity_prefix_pool.add_or_get_prefix([1, 2, 3])
    assert prefix_1 is prefix_2
    

def test_prefix_pool_max_capacity():
    max_capacity = 1
    max_capacity_prefix_pool = PrefixPool(block_size=32, max_capacity=max_capacity)
    
    # Tests that new object is created because capacity limits reached,
    # but that the newly created object is equal to the old object
    prefix_1 = max_capacity_prefix_pool.add_or_get_prefix([1, 2, 3])
    prefix_2 = max_capacity_prefix_pool.add_or_get_prefix([1, 2, 3])
    assert prefix_1 is not prefix_2
    assert prefix_1 == prefix_2

    # Tests that the max capacity remains the same
    for i in range(10):
        _ = max_capacity_prefix_pool.add_or_get_prefix(list(range(i)))
        assert len(max_capacity_prefix_pool.prefixes) == max_capacity


def test_assertion_raised_with_invalid_max_capacity():
    with pytest.raises(AssertionError):
        _ = PrefixPool(32, max_capacity=-1)
    
    with pytest.raises(AssertionError):
        _ = PrefixPool(32, max_capacity=0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
