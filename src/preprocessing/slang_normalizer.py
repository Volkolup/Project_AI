class SlangNormalizer:
    def __init__(self):
        self.slang_dict = {
            'lol': 'laughing out loud', 'omg': 'oh my god', 'omfg': 'oh my god',
            'tbh': 'to be honest', 'imo': 'in my opinion', 'imho': 'in my humble opinion',
            'btw': 'by the way', 'fyi': 'for your information', 'brb': 'be right back',
            'afk': 'away from keyboard', 'gtg': 'got to go', 'idk': 'i do not know',
            'ikr': 'i know right', 'thx': 'thanks', 'ty': 'thank you',
            'pls': 'please', 'plz': 'please', 'sry': 'sorry',
            'ur': 'your', 'u': 'you', 'r': 'are',
            'bc': 'because', 'cuz': 'because', 'cause': 'because',
            'wanna': 'want to', 'gonna': 'going to', 'gotta': 'got to',
            'dunno': 'do not know', 'kinda': 'kind of', 'sorta': 'sort of',
            'prolly': 'probably', 'def': 'definitely', 'tho': 'though',
            'rn': 'right now', 'asap': 'as soon as possible', 'nvm': 'never mind',
            'smh': 'shaking my head', 'fml': 'my life', 'wtf': 'what',
            'af': 'very', 'tfw': 'that feeling when', 'mfw': 'my face when',
            'yolo': 'you only live once', 'bae': 'babe', 'squad': 'group'
        }
    
    def normalize_text(self, text):
        words = text.lower().split()
        normalized = []
        for word in words:
            normalized.append(self.slang_dict.get(word, word))
        return ' '.join(normalized)
