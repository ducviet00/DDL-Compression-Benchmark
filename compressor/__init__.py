from .topk import TopKCompressor
from .randomk import RandomKCompressor
from .structured import StructuredCompressor
from .randomblock import RandomBlockCompressor
from .redsync import RedSyncCompressor
from .none import NoneCompressor
from .gaussiank import GaussianCompressor       
from .submaxlayer import SubMaxbyLayerCompressor

compressors = {
        'submax_by_layer': SubMaxbyLayerCompressor,
        'structured': StructuredCompressor,
        'randomblock': RandomBlockCompressor,
        'topk': TopKCompressor,
        # 'topk2': TopKCompressor2,
        'gaussian': GaussianCompressor,
        # 'gaussian2': GaussianCompressor2,
        'randomk': RandomKCompressor,
        # 'randomkec': RandomKECCompressor,
        # 'dgcsampling': DGCSamplingCompressor,
        'redsync': RedSyncCompressor,
        # 'redsynctrim': RedSyncTrimCompressor,
        'none': NoneCompressor,
        None: NoneCompressor
        }