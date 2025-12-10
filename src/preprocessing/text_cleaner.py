import re

class TextCleaner:
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s]')
    
    def clean_text(self, text):
        text = str(text)
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub(r'\1', text)
        text = self.special_chars_pattern.sub('', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text
