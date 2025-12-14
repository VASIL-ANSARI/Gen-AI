from deep_translator import GoogleTranslator

def translate(text, source, target):
    if source == target:
        return text
    return GoogleTranslator(source=source, target=target).translate(text)
