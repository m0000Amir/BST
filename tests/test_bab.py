from main.bab import BST
from main.input import placement, gateway_placement, sta


def test_noncov_inrange(place1=1, place2=10, cov1=10, cov2=15):
    bst = BST(placement, gateway_placement, sta)
    noncov = bst.noncov_inrange(place1, place2, cov1, cov2)
    assert noncov is not None
