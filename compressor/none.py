

class NoneCompressor():
    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        return tensor, None, None
    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 