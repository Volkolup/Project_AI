import emoji
import json

class EmojiProcessor:
    def __init__(self):
        self.emoji_dict = {
            '😊': 'happy', '😃': 'happy', '😀': 'happy', '😁': 'happy', '😄': 'happy',
            '😢': 'sad', '😭': 'sad', '😔': 'sad', '😞': 'sad', '😟': 'sad',
            '😡': 'angry', '😠': 'angry', '😤': 'angry', '🤬': 'angry',
            '😍': 'love', '🥰': 'love', '😘': 'love', '❤️': 'love', '💕': 'love',
            '😂': 'joy', '🤣': 'joy', '😆': 'joy',
            '😮': 'surprised', '😲': 'surprised', '😯': 'surprised',
            '😐': 'neutral', '😑': 'neutral', '😶': 'neutral',
            '👍': 'approve', '👎': 'disapprove', '🙏': 'grateful',
            '🔥': 'fire', '⭐': 'star', '💯': 'perfect',
            '😎': 'cool', '🤔': 'thinking', '😴': 'tired',
            '🙄': 'annoyed', '😏': 'smirk', '😌': 'relieved'
        }
    
    def process_text(self, text):
        for emj, description in self.emoji_dict.items():
            if emj in text:
                text = text.replace(emj, f' {description} ')
        text = emoji.replace_emoji(text, '')
        return text
