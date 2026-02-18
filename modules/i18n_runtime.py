import json
import os


_language_config_file = None
_translations = {}
_default_language = "es"
_default_audio_threshold = 0.15
_current_language = "es"
_audio_confidence_threshold = 0.15


def initialize_i18n(language_config_file, translations, default_language="es", default_audio_threshold=0.15):
    global _language_config_file, _translations, _default_language, _default_audio_threshold
    global _current_language, _audio_confidence_threshold
    _language_config_file = language_config_file
    _translations = translations
    _default_language = default_language
    _default_audio_threshold = default_audio_threshold
    _current_language = default_language
    _audio_confidence_threshold = default_audio_threshold


def get_current_language():
    return _current_language


def get_audio_confidence_threshold():
    return _audio_confidence_threshold


def cargar_idioma():
    """Carga el idioma guardado o retorna el por defecto."""
    global _current_language
    if not _language_config_file:
        return _default_language
    if os.path.exists(_language_config_file):
        try:
            with open(_language_config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                lang = config.get("language", _default_language)
                if lang in _translations:
                    _current_language = lang
                    return lang
        except Exception as e:
            print(f"Error al cargar idioma: {e}")
    _current_language = _default_language
    return _default_language


def guardar_idioma(lang):
    """Guarda el idioma seleccionado."""
    global _current_language
    if not _language_config_file:
        return False
    try:
        config = {}
        if os.path.exists(_language_config_file):
            try:
                with open(_language_config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass
        config["language"] = lang
        with open(_language_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        _current_language = lang
        return True
    except Exception as e:
        print(f"Error al guardar idioma: {e}")
        return False


def cargar_audio_threshold():
    """Carga el umbral de confianza de audio desde la configuración."""
    global _audio_confidence_threshold
    if not _language_config_file:
        return _default_audio_threshold
    if os.path.exists(_language_config_file):
        try:
            with open(_language_config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                threshold = config.get("audio_confidence_threshold", _default_audio_threshold)
                if 0.01 <= threshold <= 1.0:
                    _audio_confidence_threshold = threshold
                else:
                    _audio_confidence_threshold = _default_audio_threshold
        except Exception as e:
            print(f"Error al cargar umbral de audio: {e}")
            _audio_confidence_threshold = _default_audio_threshold
    else:
        _audio_confidence_threshold = _default_audio_threshold
    return _audio_confidence_threshold


def guardar_audio_threshold(threshold):
    """Guarda el umbral de confianza de audio."""
    global _audio_confidence_threshold
    if not _language_config_file:
        return False
    try:
        config = {}
        if os.path.exists(_language_config_file):
            try:
                with open(_language_config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass
        config["audio_confidence_threshold"] = threshold
        with open(_language_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        _audio_confidence_threshold = threshold
        return True
    except Exception as e:
        print(f"Error al guardar umbral de audio: {e}")
        return False


def translate_for_language(language_code, key, *args):
    lang_dict = _translations.get(language_code, _translations.get(_default_language, {}))
    translation = lang_dict.get(key, key)
    if args:
        try:
            return translation.format(*args)
        except Exception:
            return translation
    return translation


def t(key, *args):
    """Obtiene la traducción de una clave. Soporta formato con argumentos."""
    return translate_for_language(_current_language, key, *args)
