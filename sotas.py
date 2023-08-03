from .graph_based.aknng import AKNNG
from .graph_based.centered_knng import centeredKNNG
from .graph_based.mknng import MKNNG
from .graph_based.knng import KNNG


from .adaptive_knn.pl_nn.pl_nn import PlNearestNeighbors
from .adaptive_knn.m_knn.m_knn import MKNearestNeighbors
from .adaptive_knn.lvknn import LVKNN

__k = 10
__k_max = 15
__k_min = 5

def get_sota_models():
    models = {
        'knng': KNNG(k=__k),
        'mknng': MKNNG(k=__k_max),
        'aknng': AKNNG(k=__k_max),
        'maknng': AKNNG(k=__k_max, method='MAKNNG'),
        'centered_knng': centeredKNNG(k=__k),
        'plnn': PlNearestNeighbors(),
        'smknn': MKNearestNeighbors(mode='smknn'),
        'lmknn': MKNearestNeighbors(mode='lmknn'),
        'lvknn': LVKNN(k_min=__k_min, k_max=__k_max)
    }
    return models