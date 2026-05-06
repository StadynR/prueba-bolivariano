"""
Configuración global de pytest.
Establece variables de entorno mínimas ANTES de que los módulos de la app
sean importados, evitando que Pydantic Settings falle por falta de API key.
"""
import os

# Dummy API key para tests — nunca se usa en llamadas reales (todo mockeado)
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-tests")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma_test")
